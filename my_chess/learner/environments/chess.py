from my_chess.learner.environments import (
    Environment,
    TerminateIllegalWrapper,
    AssertOutOfBoundsWrapper,
    OrderEnforcingWrapper,
)

from typing import Literal, Dict
from chess import Board, BLACK, WHITE

from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils

import torch

def chess_env(**kwargs):
    env = chess_v6.raw_env(**kwargs)
    env = TerminateIllegalWrapper(env, illegal_reward=-1)
    env = AssertOutOfBoundsWrapper(env)
    env = OrderEnforcingWrapper(env)
    return env

class Chess(Environment):
    def __init__(
            self,
            env=None,
            render_mode:Literal[None, "human", "ansi", "rgb_array"] = None) -> None:
        assert render_mode in {None, "human", "ansi", "rgb_array"}
        env = chess_env if env is None else env
        super().__init__(env, render_mode=render_mode)

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