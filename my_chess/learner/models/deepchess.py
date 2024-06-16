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

from typing import Dict, List, Union, Type, Tuple, Literal
from pathlib import Path
from functools import cmp_to_key, partial
from itertools import product
from collections import OrderedDict
import time

from my_chess.learner.models import ModelRLLIB, ModelRRLIBConfig, Model, ModelConfig
from my_chess.learner.environments import Chess


class DeepChessFEConfig(ModelConfig):
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
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = ([DeepChessFEConfig.ACTIVATIONS[a] for a in activations]
                            if isinstance(activations, list)
                            else [DeepChessFEConfig.ACTIVATIONS[activations] for i in range(len(hidden_dims)-1)])
        self.batch_norm = batch_norm
        
    def __str__(self) -> str:
        return "Shape<{}>".format(self.hidden_dims)

class DeepChessFE(Model):
    def __init__(
        self,
        input_sample,
        config:DeepChessFEConfig = None) -> None:
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
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
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
        # postprocess.append(nn.Softmax(-1))
        return nn.Sequential(
            OrderedDict([('body',nn.Sequential(*dec)),
                         ('postprocess',nn.Sequential(*postprocess))]))

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

    def max_player(self):
        return int(self.curr_player == chess.BLACK)
    
    def min_player(self):
        return int(self.curr_player == chess.WHITE)
    
    def compare_boards(self, obs1, obs2):
        comp = self.be(torch.stack([obs1, obs2],dim=-4))
        return comp[0] - comp[1]
    
    def parse_compare(self, comp1, comp2, board_key):
        return self.compare_boards(self.heur_obs[board_key][comp1[0]], self.heur_obs[board_key][comp2[0]])
    
    def get_move_argsort(self, moves, board_key, best_to_worst=True):
        idxs = None
        if self.move_sort == 'none':
            idxs = torch.arange(len(moves))
        elif self.move_sort == 'random':
            idxs = torch.randperm(len(moves))
        elif self.move_sort == 'evaluation':
            obs = []
            for i, m1 in enumerate(moves):
                for m2 in moves[i+1:]:
                    obs.append(torch.stack([self.heur_obs[board_key][m1], self.heur_obs[board_key][m2]],dim=-4))
            obs = torch.stack(obs, dim=-5)
            heurs = self.be(obs)
            heurs_resolved = heurs[...,0]-heurs[...,1]

            mask = torch.triu(torch.ones((len(moves),len(moves)), dtype=bool), diagonal=1)
            comp_table_idxs = torch.cumsum(mask.reshape((-1,)),0).reshape((len(moves),len(moves)))-1
            comp_table_upr = heurs_resolved[comp_table_idxs]
            comp_table_upr[torch.logical_not(mask)] = 0
            comp_table_lwr = heurs_resolved[comp_table_idxs.T]
            comp_table_lwr[torch.logical_not(mask.T)] = 0
            comp_table = comp_table_upr-comp_table_lwr

            idxs = torch.argsort((comp_table>0).sum(-1), descending=best_to_worst)
        return idxs

    def forward(self, board:Board, input):
        self.positions_analyzed = 0
        start_time = time.clock_gettime(time.CLOCK_MONOTONIC)
        board = board.copy()
        if isinstance(input, dict):
            for k in input:
                input[k] = torch.tensor(input[k].copy())
        else:
            input = torch.tensor(input.copy())
        self.curr_player = board.turn == chess.WHITE
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


    def max_value(self, board:Board, input, alpha, beta, depth=1) -> Tuple[int, Tuple[float, TensorType]]:
        '''
        returns: action integer, (terminal value, observation tensor)
        '''
        # print(depth, end="")
        if self.terminal_test(board):
            self.positions_analyzed += 1
            return None, (self.utility(board), input['observation'])
        if depth > self.curr_depth:
            self.positions_analyzed += 1
            return None, (None, input['observation'])

        act = None
        val = None
        moves = chess_utils.legal_moves(board)
        next_boards = [self.simulate_move(board, a, self.max_player()) for a in moves]
        next_obs = [self.simulate_observation(b, input) for b in next_boards]

        board_key = str(board).replace(' ','')+str(self.max_player())+str(moves).replace(', ','.').lstrip('[').rstrip(']')
        if not board_key in self.heur_obs:
            self.heur_obs[board_key] = {}
            for a, o in zip(moves, next_obs):
                self.heur_obs[board_key][a] = o['observation']

        next_positions = list(zip(moves, next_boards, next_obs))
        next_pos_sorted_idxs = None
        if len(next_positions) > 1:
            next_pos_sorted_idxs = self.get_move_argsort(moves, board_key)
        else:
            next_pos_sorted_idxs = [0]

        for i in next_pos_sorted_idxs:
            a, b, obs = next_positions[i]
            update_val = False
            # b = self.simulate_move(board, a, self.max_player())
            # obs = self.simulate_observation(b, input)
            _, v = self.min_value(b, obs, alpha, beta, depth=depth+1)
            self.heur_obs[board_key][a] = v[1]

            comps = self.be(torch.stack([
                torch.stack([v[1], val if not val is None else torch.zeros_like(v[1])],dim=-4),
                torch.stack([v[1], beta if not beta is None else torch.zeros_like(v[1])],dim=-4),
                torch.stack([v[1], alpha if not alpha is None else torch.zeros_like(v[1])],dim=-4),
                ], dim=-5))

            if val is None:
                update_val = True
            else:
                if v[0] == 1:
                    update_val = True
                elif v[0] != 0 and v[0] != -1:
                    if comps[0][0] > comps[0][1]:
                        update_val = True
            if update_val:
                val = v[1]
                act = a

            if not beta is None and update_val:
                if comps[1][0] >= comps[1][1]:
                    # print('p', end='')
                    return act, (None, val)
            
            if alpha is None:
                alpha = val
            elif update_val and comps[2][0] > comps[2][1]:
                alpha = val
        # print()
        return act, (None, val)
    
    def min_value(self, board:Board, input, alpha, beta, depth=1) -> Tuple[int, Tuple[float, TensorType]]:
        '''
        returns: action integer, (terminal value, observation tensor)
        '''
        # print(depth, end="")
        if self.terminal_test(board):
            self.positions_analyzed += 1
            return None, (self.utility(board), input['observation'])
        if depth > self.curr_depth:
            self.positions_analyzed += 1
            return None, (None, input['observation'])

        act = None
        val = None
        moves = chess_utils.legal_moves(board)
        next_boards = [self.simulate_move(board, a, self.min_player()) for a in moves]
        next_obs = [self.simulate_observation(b, input) for b in next_boards]

        board_key = str(board).replace(' ','')+str(self.min_player())+str(moves).replace(', ','.').lstrip('[').rstrip(']')
        if not board_key in self.heur_obs:
            self.heur_obs[board_key] = {}
            for a, o in zip(moves, next_obs):
                self.heur_obs[board_key][a] = o['observation']

        next_positions = list(zip(moves, next_boards, next_obs))
        next_pos_sorted_idxs = None
        if len(next_positions) > 1:
            next_pos_sorted_idxs = self.get_move_argsort(moves, board_key, best_to_worst=False)
        else:
            next_pos_sorted_idxs = [0]

        for i in next_pos_sorted_idxs:
            a, b, obs = next_positions[i]
            update_val = False
            # b = self.simulate_move(board, a, self.min_player())
            # obs = self.simulate_observation(b, input)
            _, v = self.max_value(b, obs, alpha, beta, depth=depth+1)
            self.heur_obs[board_key][a] = v[1]

            comps = self.be(torch.stack([
                torch.stack([v[1], val if not val is None else torch.zeros_like(v[1])],dim=-4),
                torch.stack([v[1], alpha if not alpha is None else torch.zeros_like(v[1])],dim=-4),
                torch.stack([v[1], beta if not beta is None else torch.zeros_like(v[1])],dim=-4),
                ], dim=-5))

            if val is None:
                update_val = True
            else:
                if v[0] == -1:
                    update_val = True
                elif v[0] != 0 and v[0] != 1:
                    if comps[0][0] < comps[0][1]:
                        update_val = True
            if update_val:
                val = v[1]
                act = a

            if not alpha is None and update_val:
                if comps[1][0] <= comps[1][1]:
                    # print('p', end='')
                    return act, (None, val)

            if beta is None:
                beta = val
            elif update_val and comps[2][0] < comps[2][1]:
                beta = val
        # print()
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