from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import TensorType

import gymnasium as gym
import torch
from torch import nn, Tensor
import numpy as np
from pettingzoo.classic.chess import chess_utils
from chess import Board
import chess

from typing import Dict, List, Union, Type, Tuple, Literal, Callable
from typing_extensions import override
from pathlib import Path
from functools import cmp_to_key, partial
from itertools import product
from collections import OrderedDict
import time

from my_chess.learner.models import ModelRLLIB, ModelRRLIBConfig, Model, ModelConfig, ModelAutoEncodable
from my_chess.learner.environments import Chess


class DeepChessFEConfig(ModelConfig):
    """
    Governs the infrastructure of the DeepChess geature extractor model.
        
    See :py:class:`DeepChessFE` for detail on model inner workings.
    """
    ACTIVATIONS = {
        'relu':nn.ReLU,
        'sigmoid':nn.Sigmoid
    }
    def __init__(
            self,
            hidden_dims:Union[int, List[int]]=[4096, 1024, 256, 128],
            activations:Union[str, List[str]]='relu',
            batch_norm:bool = True
            ) -> None:
        """
        Args:
            hidden_dims: Number of features at each layer of a fully connected
              neural network.
            activations: The activation function to be used after each hidden
              layer in the network. Can be customized for each layer or the same
              activation for all layers.
            batch_norm: Whether batch normalization should be applied between
              each hidden layer.
        """
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = ([DeepChessFEConfig.ACTIVATIONS[a] for a in activations]
                            if isinstance(activations, list)
                            else [DeepChessFEConfig.ACTIVATIONS[activations] for i in range(len(hidden_dims)-1)])
        self.batch_norm = batch_norm
        
    def __str__(self) -> str:
        return "Shape<{}>".format(self.hidden_dims)

class DeepChessFE(ModelAutoEncodable):
    """
    Feature extractor portion of DeepChess model.

    DeepChess begins with a feature extractor applied to a matrix representation
    of a chess board. The model flattens the height, width, and channels
    supplied by input. Then, applies several layers of fully connected linear
    layers, each followed by an activation function. Optionally, a batch
    normalization can be applied after activation, which may help with
    propagation of the learning gradient. This all is with the aim of distilling
    the input data into set of features descriptive of the board.
    """
    def __init__(
        self,
        input_sample,
        config:DeepChessFEConfig = None) -> None:
        """
        Args:
            input_sample: A sample of the input training data. This is used to 
              appropriately structure the model to dynamically accept inputs.
            config: Configration of feature extractor.
        """
        super().__init__()
        self.config = config
        self.input_shape = input_sample.shape[-3:]
        ff = []
        for i, lyr_dim in enumerate(self.config.hidden_dims):
            lyr = []
            post_processing = []
            if i == 0:
                lyr.append(nn.Linear(self.input_shape.numel(), lyr_dim))
            else:
                lyr.append(nn.Linear(self.config.hidden_dims[i-1], lyr_dim))
            if i < len(self.config.hidden_dims)-1:
                if self.config.batch_norm:
                    post_processing.append(nn.BatchNorm1d(lyr_dim))
                post_processing.append(self.config.activations[i]())
            lyr.append(nn.Sequential(*post_processing))
            ff.append(nn.Sequential(*lyr))
        self.preprocess = nn.Flatten(-3)
        self.body = nn.Sequential(*ff)
        
    @override
    def __getitem__(self, key):
        return nn.Sequential(self.preprocess, *self.body[key])
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        """
        
        Args:
          input: Training data, batched or unbatched.
        """
        if len(input.shape) < 4:
            input = input.unsqueeze(0)
        flt = self.preprocess(input)
        logits = self.body(flt)
        return logits
    
    def decoder(self):
        dec = []
        postprocess = []
        hid_dims = list(reversed(self.config.hidden_dims))
        actvs = list(reversed(self.config.activations))
        for i, lyr_dim in enumerate(hid_dims):
            lyr = []
            post_processing = []
            if i == len(hid_dims)-1:
                lyr.append(nn.Linear(lyr_dim, self.input_shape.numel()*2))
            else:
                lyr.append(nn.Linear(lyr_dim, hid_dims[i+1]))
            if i < len(hid_dims)-1:
                if self.config.batch_norm:
                    post_processing.append(nn.BatchNorm1d(hid_dims[i+1]))
                post_processing.append(actvs[i]())
            lyr.append(nn.Sequential(*post_processing))
            dec.append(nn.Sequential(*lyr))
        postprocess.append(nn.Unflatten(-1, (*self.input_shape, 2)))
        class Decoder(nn.Module):
            def __init__(slf) -> None:
                super().__init__()
                slf.body = nn.Sequential(*dec)
                slf.postprocess = nn.Sequential(*postprocess)
            def __getitem__(slf, key):
                return nn.Sequential(*slf.body[key], slf.postprocess)
            def forward(slf, x):
                slf.to(device=x.device)
                return slf.postprocess(slf.body(x))

        return Decoder()

class DeepChessEvaluatorConfig(ModelConfig):
    ACTIVATIONS = {
        'relu':nn.ReLU,
        'sigmoid':nn.Sigmoid
    }
    def __init__(
            self,
            feature_extractor:Type[Model]=None,
            feature_extractor_config:ModelConfig=None,
            feature_extractor_param_dir:Union[str, Path]=None,
            hidden_dims:Union[int, List[int]]=[512, 252, 128],
            activations:Union[str, List[str]]='relu',
            batch_norm:bool=True,
            ) -> None:
        super().__init__()
        self.feature_extractor:Type[Model] = feature_extractor
        self.feature_extractor_config = feature_extractor_config
        self.feature_extractor_param_dir = feature_extractor_param_dir
        self.hidden_dims = hidden_dims
        self.activations = ([DeepChessFEConfig.ACTIVATIONS[a] for a in activations]
                            if isinstance(activations, list)
                            else [DeepChessFEConfig.ACTIVATIONS[activations] for i in range(len(hidden_dims))])
        self.batch_norm = batch_norm
        
    def __str__(self) -> str:
        return "Shape<{}>".format(self.hidden_dims)

class DeepChessEvaluator(Model):
    def __init__(
        self,
        input_sample,
        config:DeepChessEvaluatorConfig = None) -> None:
        super().__init__()
        self.config = config
        self.flatten = nn.Flatten(1)
        self.fe = self.config.feature_extractor(input_sample=input_sample, config=self.config.feature_extractor_config)
        if self.config.feature_extractor_param_dir:
            self.fe.load_state_dict(torch.load(self.config.feature_extractor_param_dir, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        fe_params = next(iter(self.fe.parameters()))
        input_sample = input_sample.to(dtype=fe_params.dtype, device=fe_params.device)
        input_sample = input_sample[...,0,:,:,:]
        self.fe.eval()
        sample_fe_output = self.flatten(self.fe(input_sample))
        ff = []
        for i, lyr_dim in enumerate(self.config.hidden_dims):
            if i == 0:
                ff.append(nn.Linear(sample_fe_output.shape[-1:].numel()*2, lyr_dim))
            else:
                ff.append(nn.Linear(self.config.hidden_dims[i-1], lyr_dim))
            # if i < len(self.config.hidden_dims)-1:
            if self.config.batch_norm:
                ff.append(nn.BatchNorm1d(lyr_dim))
            ff.append(self.config.activations[i]())
        ff.append(nn.Linear(self.config.hidden_dims[-1], 2))
        self.ff = nn.Sequential(*ff)
        self.probs = nn.Softmax(-1)
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        input = input.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        if len(input.shape) < 4:
            input = input.unsqueeze(0)
        feat_pos_1 = self.flatten(self.fe(input[...,0,:,:,:]))
        feat_pos_2 = self.flatten(self.fe(input[...,1,:,:,:]))
        logits = self.ff(torch.cat((feat_pos_1, feat_pos_2),-1))
        probs = self.probs(logits)
        return probs

class NextPositions:
    def __init__(
            self,
            board:Board,
            latest_observation:torch.tensor,
            turn_player:int,
            perspective_player:int,
            maximizing:bool=True,
            move_sort:Literal['none', 'random', 'evaluation'] = 'evaluation',
            evaluator:Callable = None) -> None:
        self.move_sort = move_sort
        self.evaluator = evaluator
        self.maximizing = maximizing
        self.moves = chess_utils.legal_moves(board)
        self.next_boards = []
        self.next_observations = []
        self.heuristic_observations = []
        for move in self.moves:
            next_board = Chess.simulate_move(
                board,
                move,
                turn_player)
            self.next_boards.append(next_board)
            next_obs = Chess.simulate_observation(
                next_board,
                latest_observation,
                perspective_player)
            self.next_observations.append(next_obs)
            self.heuristic_observations.append(next_obs['observation'])
        self.sorted_idxs = self.get_move_argsort(self.maximizing)
        self.current_idx = -1

    def __iter__(self):
        return self
    
    def __next__(self):
        self.current_idx += 1
        if self.current_idx < len(self.moves):
            return (
                self.moves[self.sorted_idxs[self.current_idx]],
                self.next_boards[self.sorted_idxs[self.current_idx]],
                self.next_observations[self.sorted_idxs[self.current_idx]]
                )
        else:
            raise StopIteration

    def get_move_argsort(self, best_to_worst=True):
        idxs = [0]
        if len(self.moves) > 1:
            if self.move_sort == 'none':
                idxs = torch.arange(len(self.moves))
            elif self.move_sort == 'random':
                idxs = torch.randperm(len(self.moves))
            elif self.move_sort == 'evaluation':
                combinations = torch.combinations(torch.arange(len(self.heuristic_observations)))
                obs = torch.stack(self.heuristic_observations)[combinations]
                heurs = torch.nn.Softmax(dim=-1)(self.evaluator(obs))
                heurs_resolved = heurs[...,0]-heurs[...,1]

                mask = torch.triu(torch.ones((len(self.moves),len(self.moves)), dtype=bool), diagonal=1)
                comp_table_idxs = torch.cumsum(mask.reshape((-1,)),0).reshape((len(self.moves),len(self.moves)))-1
                comp_table_upr = heurs_resolved[comp_table_idxs]
                comp_table_upr[torch.logical_not(mask)] = 0
                comp_table_lwr = heurs_resolved[comp_table_idxs.T]
                comp_table_lwr[torch.logical_not(mask.T)] = 0
                comp_table = comp_table_upr-comp_table_lwr

                idxs = torch.argsort((comp_table>0).sum(-1), descending=best_to_worst)
        return idxs

class NextPositionsGenerator:
    """
    Controls generation of next positions for MiniMax Search.

    This includes how they will be sorted and making use of a hash table.
    """
    def __init__(
            self,
            move_sort:Literal['none', 'random', 'evaluation'] = 'evaluation',
            evaluator:Callable = None):
        self.move_sort = move_sort
        self.evaluator = evaluator
        if self.evaluator is None:
            assert self.move_sort != 'evaluation'
    
    def generate_next_positions(self,
            board:Board,
            latest_observation:torch.tensor,
            turn_player:int,
            perspective_player:int,
            maximizing:bool=True) -> None:
        return NextPositions(
            board = board,
            latest_observation = latest_observation,
            turn_player = turn_player,
            perspective_player = perspective_player,
            maximizing = maximizing,
            move_sort = self.move_sort,
            evaluator = self.evaluator,)


class DeepChessAlphaBetaConfig(ModelConfig):
    def __init__(
            self,
            board_evaluator:Type[Model]=None,
            board_evaluator_config:ModelConfig=None,
            board_evaluator_param_dir:Union[str, Path]=None,
            max_depth:int = 8,
            iterate_depths:bool = True,
            move_sort:Literal['none', 'random', 'evaluation'] = 'evaluation'
            ) -> None:
        super().__init__()
        self.board_evaluator:Type[Model] = board_evaluator
        self.board_evaluator_config = board_evaluator_config
        self.board_evaluator_param_dir = board_evaluator_param_dir
        self.max_depth = max_depth
        self.iterate_depths = iterate_depths
        self.move_sort = move_sort
    
class DeepChessAlphaBeta(Model):
    WHITE, BLACK = [False, True] # to match pettingzoo, opposite "chess" library
    def __init__(
        self,
        input_sample:Union[TensorType, Dict],
        config:DeepChessAlphaBetaConfig = None) -> None:
        super().__init__()
        self.config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(input_sample, dict):
            input_sample = input_sample['observation']
        input_sample = torch.stack([torch.tensor(input_sample), torch.tensor(input_sample)], -4)
        self.be = self.config.board_evaluator(input_sample=input_sample, config=self.config.board_evaluator_config)
        if self.config.board_evaluator_param_dir:
            self.be.load_state_dict(torch.load(self.config.board_evaluator_param_dir))
        self.be.to(device=device)
        self.be.eval()
        # self.be = lambda x: torch.ones(x.shape[0])[:,None]*torch.tensor([1,0])
        self.max_depth = config.max_depth
        self.iter_depths = config.iterate_depths
        self.curr_depth = 1
        self.heur_obs = {}
        self.curr_player = None
        self.positions_analyzed = 0
        self.move_sort = self.config.move_sort
        self.next_pos_gen = NextPositionsGenerator(
            move_sort=self.move_sort,
            evaluator=self.be)

    def max_player(self):
        return int(self.curr_player == DeepChessAlphaBeta.BLACK)
    
    def min_player(self):
        return int(self.curr_player == DeepChessAlphaBeta.WHITE)
    
    def update_curr_player(self, board:Board):
        self.curr_player = board.turn == DeepChessAlphaBeta.WHITE
    
    def compare_boards(self, obs1, obs2):
        comp = self.be(torch.stack([obs1, obs2],dim=-4))
        return comp[0] - comp[1]
    
    def parse_compare(self, comp1, comp2, board_key):
        return self.compare_boards(self.heur_obs[board_key][comp1[0]], self.heur_obs[board_key][comp2[0]])
  
    def forward(self, board:Board, input):
        self.positions_analyzed = 0
        start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        board = board.copy()
        if isinstance(input, dict):
            for k in input:
                input[k] = torch.tensor(input[k].copy())
        else:
            input = torch.tensor(input.copy())
        self.update_curr_player(board)
        act = None
        if self.iter_depths:
            act = self.iterate_depths(board, input)
        else:
            self.curr_depth = self.max_depth
            act, val = self.max_value(board, input, alpha=None, beta=None)
        end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        print("Total move time:{:.4f}\nPositions analyzed:{}".format(end_time - start_time, self.positions_analyzed))
        return act

    def iterate_depths(self, board:Board, input):
        act = None
        self.curr_depth = 1
        self.heur_obs = {}
        while self.curr_depth <= self.max_depth:
            dep_start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
            act, val = self.max_value(board, input, alpha=None, beta=None)
            dep_end_time = time.clock_gettime(time.CLOCK_MONOTONIC)
            print("Depth {} time:{:.4f}".format(self.curr_depth, dep_end_time - dep_start_time))
            self.curr_depth += 1
        return act

    def val_act_update(
        self,
        result:Tuple[int, torch.Tensor],
        act:int,
        heur_comp:torch.Tensor,
        ret_val:Tuple[int, torch.Tensor],
        ret_act:int,
        minimize:bool = True):
        if minimize:
            if not result[0] is None:
                result = (result[0]*-1, result[1])
            heur_comp = heur_comp * -1
        update_val = (
            ret_val is None or
            result[0] == 1 or
            (result[0] != 0 and result[0] != -1 and heur_comp[0] > heur_comp[1])
        )
        if update_val:
            ret_val = result[1]
            ret_act = act
        return ret_val, ret_act

    def return_early(
        self,
        heur_comp:torch.Tensor,
        alpha:torch.Tensor,
        beta:torch.Tensor,
        minimize:bool = True):
        ret_determiner = beta
        if minimize:
            ret_determiner = alpha
            heur_comp = heur_comp * -1
        return (
            not ret_determiner is None and
            heur_comp[0] >= heur_comp[1]
            )
    
    def alpha_beta_update(
        self,
        heur_comp:torch.Tensor,
        ret_val:Tuple[int, torch.Tensor],
        alpha:torch.Tensor,
        beta:torch.Tensor,
        minimize:bool = True):
        ret = {
            'alpha':alpha,
            'beta':beta
        }
        update_val = 'alpha'
        if minimize:
            update_val = 'beta'
            heur_comp = heur_comp * -1

        if (
            ret[update_val] is None or
            heur_comp[0] > heur_comp[1]
            ):
            ret[update_val] = ret_val
            alpha = ret_val
        return (ret['alpha'], ret['beta'])
    
    def update_values(
        self,
        result:Tuple[int, torch.Tensor],
        act:int,
        ret_val:Tuple[int, torch.Tensor],
        ret_act:int,
        beta:torch.Tensor,
        alpha:torch.Tensor,
        minimize:bool = True):
        _, v = result
        comps = self.be(torch.stack([
            torch.stack([v[1], ret_val if not ret_val is None else torch.zeros_like(v[1])],dim=-4),
            torch.stack([v[1], beta if not beta is None else torch.zeros_like(v[1])],dim=-4),
            torch.stack([v[1], alpha if not alpha is None else torch.zeros_like(v[1])],dim=-4),
            ], dim=-5))
        
        ret_val, ret_act = self.val_act_update(
            result=v,
            act=act,
            heur_comp=comps[0],
            ret_val=ret_val,
            ret_act=ret_act,
            minimize=minimize
        )

        if self.return_early(
            heur_comp=comps[1],
            alpha=alpha,
            beta=beta,
            minimize=minimize
        ):
            return True, ret_val, ret_act, alpha, beta
        
        alpha, beta = self.alpha_beta_update(
            heur_comp=comps[0],
            ret_val=ret_val,
            alpha=alpha,
            beta=beta,
            minimize=minimize
        )

        return False, ret_val, ret_act, alpha, beta

    def max_value(self, board:Board, input, alpha, beta, depth=1) -> Tuple[int, Tuple[float, TensorType]]:
        '''
        returns: action integer, (terminal value, observation tensor)
        '''
        if self.terminal_test(board):
            self.positions_analyzed += 1
            return None, (self.utility(board), input['observation'])
        if depth > self.curr_depth:
            self.positions_analyzed += 1
            return None, (None, input['observation'])

        act = None
        val = None
        next_positions = self.next_pos_gen.generate_next_positions(
            board=board,
            latest_observation=input,
            turn_player=self.max_player(),
            perspective_player=self.min_player())
        
        for a, b, obs in next_positions:
            result = self.min_value(b, obs, alpha, beta, depth=depth+1)
            #TODO add heuristic obs to next_pos_gen?
            # self.heur_obs[board_key][a] = v[1]
            ret_early, val, act, alpha, beta = self.update_values(
                result=result,
                act=a,
                ret_val=val,
                ret_act=act,
                beta=beta,
                alpha=alpha,
                minimize=False
            )
            if ret_early:
                break
        return act, (None, val)
    
    def min_value(self, board:Board, input, alpha, beta, depth=1) -> Tuple[int, Tuple[float, TensorType]]:
        '''
        returns: action integer, (terminal value, observation tensor)
        '''
        if self.terminal_test(board):
            self.positions_analyzed += 1
            return None, (self.utility(board), input['observation'])
        if depth > self.curr_depth:
            self.positions_analyzed += 1
            return None, (None, input['observation'])

        act = None
        val = None
        next_positions = self.next_pos_gen.generate_next_positions(
            board=board,
            latest_observation=input,
            turn_player=self.min_player(),
            perspective_player=self.max_player())
        
        for a, b, obs in next_positions:
            result = self.max_value(b, obs, alpha, beta, depth=depth+1)
            #TODO add heuristic obs to next_pos_gen?
            # self.heur_obs[board_key][a] = v[1]
            ret_early, val, act, alpha, beta = self.update_values(
                result=result,
                act=a,
                ret_val=val,
                ret_act=act,
                beta=beta,
                alpha=alpha,
                minimize=True
            )
            if ret_early:
                break
        return act, (None, val)

    def terminal_test(self, board:Board):
        return board.is_game_over(claim_draw=True)
    
    def utility(self, board:Board):
        black_modifier = -1 if self.curr_player == DeepChessAlphaBeta.BLACK else 1
        return chess_utils.result_to_int(board.result(claim_draw=True)) * black_modifier
    
    # def simulate_move(self, board:Board, action, player):
    #     board = board.copy()
    #     move = chess_utils.action_to_move(board, action, player)
    #     board.push(move)
    #     return board
    
    # def simulate_observation(self, board, input):
    #     """Taken from Petting Zoo Chess Environment - modified as needed"""
    #     board_history = input["observation"]

    #     observation = chess_utils.get_observation(board, self.max_player())
    #     observation = np.dstack((observation, board_history[:, :, 7:-13]))
    #     legal_moves = chess_utils.legal_moves(board)
    #     action_mask = torch.zeros(4672, dtype=torch.int8)
    #     action_mask[torch.tensor(legal_moves)] = 1

    #     return {"observation": torch.tensor(observation), "action_mask": action_mask}
    
class DeepChessRLConfig(ModelConfig):
    ACTIVATIONS = {
        'relu':nn.ReLU
    }
    def __init__(
            self,
            feature_extractor:Type[Model]=None,
            feature_extractor_config:ModelConfig=None,
            feature_extractor_param_dir:Union[str, Path]=None,
            hidden_dims:Union[int, List[int]]=[512, 252, 128],
            activations:Union[str, List[str]]='relu'
            ) -> None:
        super().__init__()
        self.feature_extractor:Type[Model] = feature_extractor
        self.feature_extractor_config = feature_extractor_config
        self.feature_extractor_param_dir = feature_extractor_param_dir
        self.hidden_dims = hidden_dims
        self.activations = ([DeepChessFEConfig.ACTIVATIONS[a] for a in activations]
                            if isinstance(activations, list)
                            else [DeepChessFEConfig.ACTIVATIONS[activations] for i in range(len(hidden_dims))])
        
    def __str__(self) -> str:
        return "Shape<{}>".format(self.hidden_dims)

class DeepChessRL(ModelRLLIB):
    def __init__(
        self,
        obs_space: gym.spaces.Space=None,
        action_space: gym.spaces.Space=None,
        num_outputs: int=None,
        model_config: ModelConfigDict=None,
        name: str=None,
        config:DeepChessRLConfig = None,
        **kwargs) -> None:
        super().__init__(
            obs_space = obs_space,
            action_space = action_space,
            num_outputs = num_outputs,
            model_config = model_config,
            name = name
            )
        self.config = config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        orig_space = getattr(self.obs_space,"original_space",self.obs_space)
        input_sample = torch.tensor(orig_space.sample()['observation'])
        self.fe = self.config.feature_extractor(input_sample=input_sample, config=self.config.feature_extractor_config)
        if self.config.feature_extractor_param_dir:
            self.fe.load_state_dict(torch.load(self.config.feature_extractor_param_dir, map_location=device))

        fe_params = next(iter(self.fe.parameters()))
        input_sample = input_sample.to(dtype=fe_params.dtype, device=fe_params.device)
        sample_fe_output = self.fe(input_sample)
        ff = []
        for i, lyr_dim in enumerate(self.config.hidden_dims):
            if i == 0:
                ff.append(nn.Linear(sample_fe_output.shape[-3:].numel(), lyr_dim))
            else:
                ff.append(nn.Linear(self.config.hidden_dims[i-1], lyr_dim))
            # if i < len(self.config.hidden_dims)-1:
            ff.append(self.config.activations[i]())
        ff.append(nn.Linear(self.config.hidden_dims[-1], action_space.n))
        self.ff = nn.Sequential(*ff)
        self.probs = nn.Softmax(-1)
        self.to(device=device)

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        action_mask = input_dict['obs']['action_mask']
        obs = input_dict["obs"]["observation"]
        input = obs.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        feat_pos_1 = self.fe(input)
        self._features = self.ff(feat_pos_1)
        masked_output = torch.where(
            action_mask.bool(),
            self._features,
            torch.ones_like(action_mask)*float("-inf"))
        if masked_output.isinf().all(-1).any():
            masked_output[masked_output.isinf().all(-1)] = 1/action_mask.shape[-1]
        # probs = self.probs(torch.where(
        #     action_mask.bool(),
        #     self._features,
        #     torch.ones_like(action_mask)*float("-inf")))
        # if probs.isnan().all(-1).any():
        #     probs[probs.isnan().all(-1)] = 1/action_mask.shape[-1]
        return masked_output, state

    def value_function(self):
        # return self.ff.value_function()
        return self._features.max(-1)[0]