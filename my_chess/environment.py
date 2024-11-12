from ml_training_suite.environments import (
    Environment,
    TerminateIllegalWrapper,
    AssertOutOfBoundsWrapper,
    OrderEnforcingWrapper,
)

from typing import (
    Literal,
    Dict,
    Union,
    Type
    )
from chess import Board, BLACK, WHITE

from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils
from pettingzoo.utils.env import AECEnv

import torch
import numpy as np


"""
chess_v6 env does not provide an initial observation of the board with no pieces moved.
"""
class raw_env(chess_v6.raw_env):
    def __init__(self, render_mode: Union[str,None] = None, screen_height: Union[int,None] = 800):
        super().__init__(render_mode, screen_height)
        self.board_history = np.dstack(
            (self.init_board_obs(), self.board_history[:, :, :-13])
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.board_history = np.dstack(
            (self.init_board_obs(), self.board_history[:, :, :-13])
        )

    def init_board_obs(self):
        return np.array([[[False, False, False, False, False, False, False, False, False,  True, False, False, False],
        [False, False, False, False, False, False, False,  True, False, False, False, False, False],
        [False, False, False, False, False, False, False, False,  True, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False,  True, False, False],
        [False, False, False, False, False, False, False, False, False, False, False,  True, False],
        [False, False, False, False, False, False, False, False,  True, False, False, False, False],
        [False, False, False, False, False, False, False,  True, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False,  True, False, False, False]],

       [[False, False, False, False, False, False,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False, False, False, False, False],
        [False, False, False, False, False, False,  True, False, False, False, False, False, False]],

       [[False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False]],

       [[False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False]],

       [[False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False]],

       [[False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False]],

       [[ True, False, False, False, False, False, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False, False, False, False, False, False]],

       [[False, False, False,  True, False, False, False, False, False, False, False, False, False],
        [False,  True, False, False, False, False, False, False, False, False, False, False, False],
        [False, False,  True, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False,  True, False, False, False, False, False, False, False, False],
        [False, False, False, False, False,  True, False, False, False, False, False, False, False],
        [False, False,  True, False, False, False, False, False, False, False, False, False, False],
        [False,  True, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False,  True, False, False, False, False, False, False, False, False, False]]])

    @classmethod
    def observation_to_fen(cls, observation:torch.Tensor, board_state=0):
        """For Visualizing purposes only. Will not fully capture moves, castling rights, etc."""
        observation = observation.int().moveaxis(-1,1)
        is_black_perspectives = [obs[4].all() for obs in observation]
        white_piece_ints = [ord('P'),ord('N'),ord('B'),ord('R'),ord('Q'),ord('K'),]
        black_piece_ints = [ord('p'),ord('n'),ord('b'),ord('r'),ord('q'),ord('k'),]
        piece_ints = torch.tensor([white_piece_ints+black_piece_ints if not black_persp else black_piece_ints+white_piece_ints for black_persp in is_black_perspectives])
        observation = observation[:, 7 + board_state * 13:19 + board_state * 13]

        #potential en passant pawns are represented on back ranks and need adjustment
        observation[:,0,3] = torch.logical_or(observation[:,0,3], observation[:,0,0])
        observation[:,0,0] = False
        observation[:,0,4] = torch.logical_or(observation[:,0,4], observation[:,0,7])
        observation[:,0,7] = False
        observation[:,6,3] = torch.logical_or(observation[:,6,3], observation[:,6,0])
        observation[:,6,0] = False
        observation[:,6,4] = torch.logical_or(observation[:,6,4], observation[:,6,7])
        observation[:,6,7] = False

        boards = (observation * piece_ints[:,:,None,None]).sum(1)

        fens = []
        for i, board in enumerate(boards):
            if is_black_perspectives[i]:
                board = board.flip(0)
            fen = ''
            for row in board:
                empty_space_count = 0
                for cell in row:
                    if cell == 0:
                        empty_space_count += 1
                    else:
                        if empty_space_count != 0:
                            fen += str(empty_space_count)
                        empty_space_count = 0
                        fen += chr(cell.item())
                if empty_space_count != 0:
                    fen += str(empty_space_count)
                fen += '/'
            fen = fen.rstrip('/') + ' w - - 0 1'
            fens.append(fen)
        return fens

def chess_env(**kwargs) -> AECEnv:
    env = raw_env(**kwargs)
    env = TerminateIllegalWrapper(env, illegal_reward=-1)
    env = AssertOutOfBoundsWrapper(env)
    env = OrderEnforcingWrapper(env)
    return env

class Chess(Environment):
    env=chess_env
    def __init__(
            self,
            render_mode:Literal[None, "human", "ansi", "rgb_array"] = None) -> None:
        assert render_mode in {None, "human", "ansi", "rgb_array"}
        super().__init__(render_mode=render_mode)

    @staticmethod
    def mirror_board_view(observation):
        '''Based on Petting Zoo Chess environment'''
        # 1. Mirror the board
        if len(observation.shape) == 3:
            # Not batched
            observation = torch.flip(observation, dims=(0,))
        elif len(observation.shape) == 4:
            # Batched
            observation = torch.flip(observation, dims=(1,))
        else:
            raise RuntimeError("Unexpected observation shape when attempting to mirror.")

        # 2. Swap the white 6 channels with the black 6 channels
        static_channels = torch.arange(7)
        white_channels = torch.arange(8)[:,None]*13+7+torch.arange(6)
        black_channels = torch.arange(8)[:,None]*13+7+6+torch.arange(6)
        rep_channels = torch.arange(8)[:,None]*13+7+6+6
        channel_swap = torch.cat((black_channels, white_channels, rep_channels), dim=-1)
        swap_idxs = torch.cat((static_channels, channel_swap.flatten()))
        return observation[..., swap_idxs]
    
    @staticmethod
    def simulate_move(board:Board, action:int, player:bool) -> Board:
        board = board.copy()
        move = chess_utils.action_to_move(board, action, player)
        board.push(move)
        return board
    
    @staticmethod
    def simulate_observation(
        board:Board,
        prev_observation:Dict[str,torch.Tensor],
        perspective_player:bool
        ) -> Dict[str, torch.Tensor]:
        WHITE, BLACK = [False, True]
        """Taken from Petting Zoo Chess Environment - modified as needed"""
        board_history = prev_observation["observation"]
        current_perspective = board_history[:,:,4].all() == BLACK
        if current_perspective != perspective_player:
            board_history = Chess.mirror_board_view(board_history)

        observation = torch.from_numpy(
            chess_utils.get_observation(board, perspective_player).copy())
        
        if perspective_player == BLACK: #bug in Petting Zoo flips obs clock for black
            observation[:,:,5] = observation[:,:,5].flip(0)

        observation = torch.dstack((observation, board_history[:, :, 7:-13]))
        legal_moves = chess_utils.legal_moves(board)
        action_mask = torch.zeros(4672, dtype=torch.int8)
        if legal_moves:
            action_mask[torch.tensor(legal_moves)] = 1

        return {"observation": observation, "action_mask": action_mask}