#IDEAS:
# -SWIN to produce a convolved interpretation of the board
# -

from typing import Dict, List, Optional, Callable, Union
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict
import gymnasium as gym

import numpy as np

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models.swin_transformer import (
    SwinTransformerBlockV2,
    PatchMergingV2,
    _log_api_usage_once,
    partial,
    Permute    
)

from my_chess.learner.models import Model, ModelConfig

# from matcher import HungarianMatcher

class PositionalEmbedder(nn.Module):
    def __init__(
            self,
            dummy_feature_maps,
            hidden_dim = 256,
            ) -> None:
        super(PositionalEmbedder, self).__init__()
        self.emb_z = nn.Embedding(len(dummy_feature_maps), hidden_dim)
        self.embs_x = nn.ModuleList([])
        self.embs_y = nn.ModuleList([])
        for lvl in dummy_feature_maps:
            self.embs_x.append(nn.Embedding(lvl.shape[-2], hidden_dim//2))
            self.embs_y.append(nn.Embedding(lvl.shape[-3], hidden_dim//2))

    def forward(
            self,
            feature_maps,
            ):
        z_pos = self.emb_z(torch.arange(len(feature_maps), device=feature_maps[0].device))
        pos = []
        for i, lvl in enumerate(feature_maps):
            x_pos = self.embs_x[i](torch.arange(lvl.shape[-2], device=lvl.device))
            y_pos = self.embs_y[i](torch.arange(lvl.shape[-3], device=lvl.device))
            lvl_pos = torch.cat([
                x_pos.unsqueeze(0).repeat(y_pos.shape[0],1,1),
                y_pos.unsqueeze(1).repeat(1,x_pos.shape[0],1)],
                dim=-1).unsqueeze(0).repeat(lvl.shape[0],1,1,1)
            lvl_pos = lvl_pos + z_pos[i]
            pos.append(lvl_pos)
        return pos
    
class FeatureProjector(nn.Module):
    def __init__(
            self,
            dummy_feature_maps,
            hidden_dim = 256,
            ) -> None:
        super(FeatureProjector, self).__init__()
        self.proj_list = nn.ModuleList([])
        for lvl in dummy_feature_maps:
            self.proj_list.append(
                nn.Sequential(
                    nn.Conv2d(lvl.shape[-1], hidden_dim, 1, 1),
                    nn.GroupNorm(32,hidden_dim)
                ))

    def forward(
            self,
            feature_maps,
            ):
        proj_feat = []
        for i, lvl in enumerate(feature_maps):            
            proj_feat.append(self.proj_list[i](lvl.moveaxis(-1,1)).moveaxis(1,-1))
        return proj_feat

class SwinFeatureExtractor(nn.Module):
    """
    Taken from pyTorch implementation, modified for use as needed.
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        input_channels: int,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMergingV2,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes
        self.downsample_layer = downsample_layer

        if block is None:
            block = SwinTransformerBlockV2
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    input_channels, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs):
        inp = obs.moveaxis(-1,1)
        features = []
        x = inp
        for i, module in enumerate(self.features):
            x = module(x)
            if i != 0 and module._get_name() != self.downsample_layer.__name__:
                features.append(x)
        return features

class ToBeNamedConfig(ModelConfig):
    def __init__(
        self,
        swin_input_channels: int = 111,
        swin_patch_size: List[int] = [1,1],
        swin_embed_dim: int = 32,
        swin_depths: List[int] = [2,2,6,2],
        swin_num_heads: List[int] = [2,4,8,16],
        swin_window_size: List[int] = [2,2],
        swin_mlp_ratio: float = 4.0,
        swin_dropout: float = 0.0,
        swin_attention_dropout: float = 0.0,
        swin_stochastic_depth_prob: float = 0.1,
        swin_num_classes: int = 1000,
        swin_norm_layer: Optional[Callable[..., nn.Module]] = None,
        swin_block: Optional[Callable[..., nn.Module]] = SwinTransformerBlockV2,
        swin_downsample_layer: Callable[..., nn.Module] = PatchMergingV2,
        hidden_dim: int = 256,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 2048,
        encoder_dropout: float = 0.1,
        encoder_activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        encoder_layer_norm_eps: float = 1e-5,
        encoder_batch_first: bool = True,
        encoder_norm_first: bool = False,
        encoder_num_layers: int = 4,
        encoder_norm = None,
        encoder_enable_nested_tensor = True,
        encoder_mask_check = True,
        embedding_dim: int = 256,) -> None:
        super().__init__()
        #Metadata
        self.swin_input_channels = swin_input_channels
        self.swin_patch_size = swin_patch_size
        self.swin_embed_dim = swin_embed_dim
        self.swin_depths = swin_depths
        self.swin_num_heads = swin_num_heads
        self.swin_window_size = swin_window_size
        self.swin_mlp_ratio = swin_mlp_ratio
        self.swin_dropout = swin_dropout
        self.swin_attention_dropout = swin_attention_dropout
        self.swin_stochastic_depth_prob = swin_stochastic_depth_prob
        self.swin_num_classes = swin_num_classes
        self.swin_norm_layer = swin_norm_layer
        self.swin_block = swin_block
        self.swin_downsample_layer = swin_downsample_layer
        self.hidden_dim = hidden_dim
        self.encoder_nhead = encoder_nhead
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.encoder_dropout = encoder_dropout
        self.encoder_activation = encoder_activation
        self.encoder_layer_norm_eps = encoder_layer_norm_eps
        self.encoder_batch_first = encoder_batch_first
        self.encoder_norm_first = encoder_norm_first
        self.encoder_num_layers = encoder_num_layers
        self.encoder_norm = encoder_norm
        self.encoder_enable_nested_tensor = encoder_enable_nested_tensor
        self.encoder_mask_check = encoder_mask_check
        self.embedding_dim = embedding_dim

class ToBeNamed(Model):
    def __init__(
        self,
        obs_space: gym.spaces.Space=None,
        action_space: gym.spaces.Space=None,
        num_outputs: int=None,
        model_config: ModelConfigDict=None,
        name: str=None,
        config:ToBeNamedConfig = None,
        **kwargs
    ):
        Model.__init__(
            self,
            obs_space = obs_space,
            action_space = action_space,
            num_outputs = num_outputs,
            model_config = model_config,
            name = name,
            )
        self.config = config
        #Model
        self.feature_extractor = SwinFeatureExtractor(
            input_channels = self.config.swin_input_channels,
            patch_size = self.config.swin_patch_size,
            embed_dim = self.config.swin_embed_dim,
            depths = self.config.swin_depths,
            num_heads = self.config.swin_num_heads,
            window_size = self.config.swin_window_size,
            mlp_ratio = self.config.swin_mlp_ratio,
            dropout = self.config.swin_dropout,
            attention_dropout = self.config.swin_attention_dropout,
            stochastic_depth_prob = self.config.swin_stochastic_depth_prob,
            num_classes = self.config.swin_num_classes,
            norm_layer = self.config.swin_norm_layer,
            block = self.config.swin_block,
            downsample_layer = self.config.swin_downsample_layer
        )

        sample_obs = torch.tensor(obs_space.original_space['observation'].sample(), dtype=torch.float32).unsqueeze(0).repeat(2,1,1,1)
        sample_feature_map = self.feature_extractor(sample_obs)

        self.feature_projector = FeatureProjector(sample_feature_map, hidden_dim = self.config.hidden_dim//2)
        self.pos_emb = PositionalEmbedder(sample_feature_map, hidden_dim = self.config.hidden_dim//2)

        # Initially perhaps only an encoder is necessary to encode the features and their positional embedding
        # Potentially, like DINO, it may make sense for the encoder to provide a "gut instinct move", which can further
        # be refined by the decoder. This would be most advantageous though if able to introduce auxilliary losses like 
        # CDN does, but unsure what ground truth could be used.
        # Could potentially use CDN for learning legal moves as well as learning tactics via puzzles. Noising may look like
        # changing the board scape in a way that doesn't affect which moves are legal or the solution to the puzzle, though
        # this seems challenging to implement/validate for any given board state, and will be impossible for some.
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = self.config.hidden_dim,
                nhead = self.config.encoder_nhead,
                dim_feedforward = self.config.encoder_dim_feedforward,
                dropout = self.config.encoder_dropout,
                activation = self.config.encoder_activation,
                layer_norm_eps = self.config.encoder_layer_norm_eps,
                batch_first = self.config.encoder_batch_first,
                norm_first = self.config.encoder_norm_first,
            ),
            num_layers = self.config.encoder_num_layers,
            norm = self.config.encoder_norm,
            enable_nested_tensor = self.config.encoder_enable_nested_tensor,
            mask_check = self.config.encoder_mask_check
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer = nn.TransformerDecoderLayer(
                d_model = self.config.embedding_dim,
                nhead = self.config.encoder_nhead,
                dim_feedforward = self.config.encoder_dim_feedforward,
                dropout = self.config.encoder_dropout,
                activation = self.config.encoder_activation,
                layer_norm_eps = self.config.encoder_layer_norm_eps,
                batch_first = self.config.encoder_batch_first,
                norm_first = self.config.encoder_norm_first,
            ),
            num_layers = self.config.encoder_num_layers,
            norm = self.config.encoder_norm
        )

        self.move_emb = nn.Embedding(self.num_outputs, self.config.embedding_dim)
        self.move_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.encoder_dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.config.encoder_dim_feedforward, self.num_outputs)
        )
        self.legal_move_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.encoder_dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.config.encoder_dim_feedforward, 2)
        )
        self.probs = nn.Softmax(-1)
        self.cross_ent = nn.CrossEntropyLoss()
        # self.matcher = HungarianMatcher()
        self._legal_moves = None
        self._query_move_proposal = None
        self._values = None

    def forward(self, input_dict, state, seq_lens):
        # Collect feature map from each stage of feature extractor
        qs = None
        obs = input_dict['obs']['observation']
        if torch.cuda.is_available() and obs.device != 'cuda':
            obs.to('cuda')

        features = self.feature_extractor(obs)
        proj_feat = self.feature_projector(features)
        pos_emb = self.pos_emb(features)
        enc_in = []
        for i in range(len(proj_feat)):
            e_i = torch.cat([proj_feat[i], pos_emb[i]], dim=-1)
            enc_in.append(e_i.flatten(-3,-2))
        enc_in = torch.cat(enc_in, dim=-2)

        hidden_state = self.encoder(enc_in)
        
        # moves = self.move_emb(torch.arange(self.num_outputs)).unsqueeze(0).repeat(hidden_state.shape[0],1,1)
        init_move_guess = self.move_head(hidden_state).argmax(-1)
        moves = self.move_emb(init_move_guess)

        dec_state = self.decoder(moves, hidden_state)

        legal_moves = self.probs(self.legal_move_head(dec_state))
        self._legal_moves = legal_moves
        legal_moves = legal_moves.argmax(-1)

        proposed_move_probs = self.probs(self.move_head(dec_state))
        # proposed_moves[torch.logical_not(legal_moves)] = float('-inf')
        proposed_move_confidence, proposed_moves = proposed_move_probs.max(-1)
        self._query_move_proposal = proposed_moves
        proposed_move_confidence[torch.logical_not(legal_moves)] = 0
        best_idx = proposed_move_confidence.argmax(-1)
        qs = proposed_move_probs[torch.arange(best_idx.shape[0], device=best_idx.device), best_idx]
        self._values = proposed_move_probs.amax((-1,-2))

        return qs, []
    
    def value_function(self) -> TensorType:
        return self._values

    def custom_loss(self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]) -> Union[TensorType, List[TensorType]]:
        # Top one in place of one hot function which only uses as many classes as show up. I believe the sample batch has all actions as false, so only one hot category exists.
        # self.cross_ent(legal_moves, (input_dict['obs']['action_mask'].unsqueeze(-1).repeat(1,1,2) == torch.arange(2).repeat(*input_dict['obs']['action_mask'].shape,1)).float())
        # self._legal_moves_loss = self.cross_ent(legal_moves, nn.functional.one_hot(input_dict['obs']['action_mask']).to(dtype=legal_moves.dtype))
        obs = restore_original_dimensions(loss_inputs['obs'], self.obs_space, tensorlib=self.framework)
        action_mask = obs['action_mask'][[np.indices(self._query_move_proposal.shape)[0], self._query_move_proposal]]

        legal_move_loss = self.cross_ent(self._legal_moves, (action_mask.unsqueeze(-1).repeat(1,1,2) == torch.arange(2, device=action_mask.device).repeat(*action_mask.shape,1)).float())
        # legal_move_loss = self.matcher()
        super_loss = super().custom_loss(policy_loss, loss_inputs)
        loss = [legal_move_loss + super_loss[0]]
        return loss