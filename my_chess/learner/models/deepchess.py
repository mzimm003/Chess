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

from typing import Dict, List, Union, Type, Tuple
from pathlib import Path

from my_chess.learner.models import ModelRLLIB, ModelRRLIBConfig, Model, ModelConfig
from my_chess.learner.environments import Chess


class DeepChessFEConfig(ModelConfig):
    ACTIVATIONS = {
        'relu':nn.ReLU
    }
    def __init__(
            self,
            hidden_dims:Union[int, List[int]]=[4096, 1024, 256, 128],
            activations:Union[str, List[str]]='relu'
            ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = ([DeepChessFEConfig.ACTIVATIONS[a] for a in activations]
                            if isinstance(activations, list)
                            else [DeepChessFEConfig.ACTIVATIONS[activations] for i in range(len(hidden_dims)-1)])
        
    def __str__(self) -> str:
        return "Shape<{}>".format(self.hidden_dims)

class DeepChessFE(Model):
    def __init__(
        self,
        input_sample,
        config:DeepChessFEConfig = None) -> None:
        super().__init__()
        self.config = config
        ff = []
        for i, lyr_dim in enumerate(self.config.hidden_dims):
            if i == 0:
                ff.append(nn.Linear(input_sample.shape[-3:].numel(), lyr_dim))
            else:
                ff.append(nn.Linear(self.config.hidden_dims[i-1], lyr_dim))
            if i < len(self.config.hidden_dims)-1:
                ff.append(self.config.activations[i]())
        self.flatten = nn.Flatten(-3)
        self.ff = nn.Sequential(*ff)
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        
        flt = self.flatten(input)
        logits = self.ff(flt)
        return logits
    
class DeepChessEvaluatorConfig(ModelConfig):
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

class DeepChessEvaluator(Model):
    def __init__(
        self,
        input_sample,
        config:DeepChessEvaluatorConfig = None) -> None:
        super().__init__()
        self.config = config
        self.fe = self.config.feature_extractor(input_sample=input_sample, config=self.config.feature_extractor_config)
        if self.config.feature_extractor_param_dir:
            self.fe.load_state_dict(torch.load(self.config.feature_extractor_param_dir))

        fe_params = next(iter(self.fe.parameters()))
        input_sample = input_sample.to(dtype=fe_params.dtype, device=fe_params.device)
        sample_fe_output = self.fe(input_sample)
        ff = []
        for i, lyr_dim in enumerate(self.config.hidden_dims):
            if i == 0:
                ff.append(nn.Linear(sample_fe_output.shape[-3:].numel()*2, lyr_dim))
            else:
                ff.append(nn.Linear(self.config.hidden_dims[i-1], lyr_dim))
            # if i < len(self.config.hidden_dims)-1:
            ff.append(self.config.activations[i]())
        ff.append(nn.Linear(self.config.hidden_dims[-1], 2))
        self.ff = nn.Sequential(*ff)
        self.probs = nn.Softmax(-1)
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        input = input.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        feat_pos_1 = self.fe(input[...,0,:,:,:])
        feat_pos_2 = self.fe(input[...,1,:,:,:])
        logits = self.ff(torch.cat((feat_pos_1, feat_pos_2),-1))
        probs = self.probs(logits)
        return probs

class DeepChessAlphaBetaConfig(ModelConfig):
    def __init__(
            self,
            board_evaluator:Type[Model]=None,
            board_evaluator_config:ModelConfig=None,
            board_evaluator_param_dir:Union[str, Path]=None,
            depth:int = 8,
            ) -> None:
        super().__init__()
        self.board_evaluator:Type[Model] = board_evaluator
        self.board_evaluator_config = board_evaluator_config
        self.board_evaluator_param_dir = board_evaluator_param_dir
        self.depth = depth
    
class DeepChessAlphaBeta(Model):
    def __init__(
        self,
        input_sample:Union[TensorType, Dict],
        config:DeepChessAlphaBetaConfig = None) -> None:
        super().__init__()
        self.config = config
        if isinstance(input_sample, dict):
            input_sample = input_sample['observation']
        input_sample = torch.tensor(input_sample)
        self.be = self.config.board_evaluator(input_sample=input_sample, config=self.config.board_evaluator_config)
        if self.config.board_evaluator_param_dir:
            self.be.load_state_dict(torch.load(self.config.board_evaluator_param_dir))
        self.depth = config.depth
        self.curr_player = None

    def max_player(self):
        return int(self.curr_player == chess.BLACK)
    
    def min_player(self):
        return int(self.curr_player == chess.WHITE)

    def forward(self, board:Board, input):
        board = board.copy()
        if isinstance(input, dict):
            for k in input:
                input[k] = torch.tensor(input[k].copy())
        else:
            input = torch.tensor(input.copy())
        self.curr_player = board.turn == chess.WHITE
        act, val = self.max_value(board, input, alpha=None, beta=None)
        return act

    def max_value(self, board:Board, input, alpha, beta, depth=1) -> Tuple[int, float]:
        if self.terminal_test(board):
            return None, (self.utility(board), input['observation'])
        if depth > self.depth:
            return None, (None, input['observation'])

        act = None
        val = None
        for a in chess_utils.legal_moves(board):
            update_val = False
            b = self.simulate_move(board, a, self.max_player())
            obs = self.simulate_observation(b, input)
            _, v = self.min_value(b, obs, alpha, beta, depth=depth+1)

            if val is None:
                update_val = True
            else:
                if v[0] == 1:
                    update_val = True
                elif v[0] != 0 and v[0] != -1:
                    comp = self.be(torch.stack([val, v[1]],dim=-4))
                    if comp[1] > comp[0]:
                        update_val = True
            if update_val:
                val = v[1]
                act = a

            if not beta is None and update_val:
                comp = self.be(torch.stack([val, beta],dim=-4))
                if comp[0] >= comp[1]:
                    return act, (None, val)
            
            if alpha is None:
                alpha = val
            else:
                comp = self.be(torch.stack([val, alpha],dim=-4))
                if comp[0] > comp[1]:
                    alpha = val
        return act, (None, val)
    
    def min_value(self, board:Board, input, alpha, beta, depth=1) -> Tuple[int, float]:
        if self.terminal_test(board):
            return None, (self.utility(board), input['observation'])
        if depth > self.depth:
            return None, (None, input['observation'])

        act = None
        val = None
        for a in chess_utils.legal_moves(board):
            update_val = False
            b = self.simulate_move(board, a, self.min_player())
            obs = self.simulate_observation(b, input)
            _, v = self.max_value(b, obs, alpha, beta, depth=depth+1)

            if val is None:
                update_val = True
            else:
                if v[0] == -1:
                    update_val = True
                elif v[0] != 0 and v[0] != 1:
                    comp = self.be(torch.stack([val, v[1]],dim=-4))
                    if comp[1] < comp[0]:
                        update_val = True
            if update_val:
                val = v[1]
                act = a

            if not alpha is None and update_val:
                comp = self.be(torch.stack([val, alpha],dim=-4))
                if comp[0] <= comp[1]:
                    return act, (None, val)
            if beta is None:
                beta = val
            else:
                comp = self.be(torch.stack([val, beta],dim=-4))
                if comp[0] < comp[1]:
                    beta = val
        return act, (None, val)

    def terminal_test(self, board:Board):
        return board.is_game_over(claim_draw=True)
    
    def utility(self, board:Board):
        black_modifier = -1 if self.curr_player == chess.BLACK else 1
        return chess_utils.result_to_int(board.result(claim_draw=True)) * black_modifier
    
    def simulate_move(self, board:Board, action, player):
        board = board.copy()
        move = chess_utils.action_to_move(board, action, player)
        board.push(move)
        return board
    
    def simulate_observation(self, board, input):
        """Taken from Petting Zoo Chess Environment - modified as needed"""
        board_history = input["observation"]

        observation = chess_utils.get_observation(board, self.max_player())
        observation = np.dstack((observation, board_history[:, :, 7:-13]))
        legal_moves = chess_utils.legal_moves(board)
        action_mask = np.zeros(4672, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": torch.tensor(observation), "action_mask": torch.tensor(action_mask)}