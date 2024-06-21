"""
"""
import pytest

import torch
from pettingzoo.classic.chess import chess_utils
from chess import Board
import chess

from my_chess.learner.models import Model, ModelConfig, DeepChessAlphaBeta, DeepChessAlphaBetaConfig
from my_chess.learner.environments import Chess

class TestDeepChessAlphaBeta:
    class DummyEvaluatorConfig(ModelConfig):
        def __init__(self, depth=10):
            super().__init__()
            self.depth = depth

    class DummyEvaluator(Model):
        def __init__(self, input_sample, config):
            super().__init__()
            self.ff = torch.nn.Sequential(*[torch.nn.Identity()]*config.depth)
        
        def forward(self, input):
            x = self.ff(input)
            return torch.ones(x.shape[0],2,device=x.device) * .5

    environment = Chess()
    input_sample, _ = environment.reset()
    input_sample = next(iter(input_sample.values()))
    model = DeepChessAlphaBeta(
        input_sample=input_sample,
        config=DeepChessAlphaBetaConfig(
            board_evaluator=DummyEvaluator,
            board_evaluator_config=DummyEvaluatorConfig(),
            max_depth=2
        )
    )
    base_observation = {"observation": torch.cat([
        torch.ones(8,8,7)*torch.nan,
        torch.arange(8).repeat_interleave(13).repeat(8,8,1)], dim=-1)}
    
    def test_max_player_1(self):
        self.model.update_curr_player(self.environment.env.board)
        assert self.model.max_player() == 0

    def test_max_player_2(self):
        self.environment.step({'player_0':77})
        self.model.update_curr_player(self.environment.env.board)
        assert self.model.max_player() == 1

    def test_min_player_1(self):
        self.model.update_curr_player(self.environment.env.board)
        assert self.model.min_player() == 1

    def test_min_player_2(self):
        self.environment.step({'player_0':77})
        self.model.update_curr_player(self.environment.env.board)
        assert self.model.min_player() == 0

    def test_simulate_move_1(self):
        board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.model.update_curr_player(board)
        next_boards = [
            Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/8/7P/PPPPPPP1/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq - 0 1'),
            Board('rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq - 0 1'),
        ]
        moves = chess_utils.legal_moves(board)
        results = [self.model.simulate_move(board, a, self.model.max_player()) for a in moves]
        for result in results:
            assert result in next_boards
            next_boards.remove(result)
        assert next_boards == []

    def test_simulate_move_2(self):
        init_board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.model.update_curr_player(init_board)
        next_next_boards_white = [
            ('8/7N/PPPPPPPP/RNBQKB1R w KQkq - {} 2', 1),
            ('8/5N2/PPPPPPPP/RNBQKB1R w KQkq - {} 2', 1),
            ('8/N7/PPPPPPPP/R1BQKBNR w KQkq - {} 2', 1),
            ('8/2N5/PPPPPPPP/R1BQKBNR w KQkq - {} 2', 1),
            ('8/P7/1PPPPPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('8/1P6/P1PPPPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('8/2P5/PP1PPPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('8/3P4/PPP1PPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('8/4P3/PPPP1PPP/RNBQKBNR w KQkq - {} 2', 0),
            ('8/5P2/PPPPP1PP/RNBQKBNR w KQkq - {} 2', 0),
            ('8/6P1/PPPPPP1P/RNBQKBNR w KQkq - {} 2', 0),
            ('8/7P/PPPPPPP1/RNBQKBNR w KQkq - {} 2', 0),
            ('P7/8/1PPPPPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('1P6/8/P1PPPPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('2P5/8/PP1PPPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('3P4/8/PPP1PPPP/RNBQKBNR w KQkq - {} 2', 0),
            ('4P3/8/PPPP1PPP/RNBQKBNR w KQkq - {} 2', 0),
            ('5P2/8/PPPPP1PP/RNBQKBNR w KQkq - {} 2', 0),
            ('6P1/8/PPPPPP1P/RNBQKBNR w KQkq - {} 2', 0),
            ('7P/8/PPPPPPP1/RNBQKBNR w KQkq - {} 2', 0),
        ]
        next_next_boards_black = [
            ('rnbqkb1r/pppppppp/7n/8', 1),
            ('rnbqkb1r/pppppppp/5n2/8', 1),
            ('r1bqkbnr/pppppppp/n7/8', 1),
            ('r1bqkbnr/pppppppp/2n5/8', 1),
            ('rnbqkbnr/1ppppppp/p7/8', 0),
            ('rnbqkbnr/p1pppppp/1p6/8', 0),
            ('rnbqkbnr/pp1ppppp/2p5/8', 0),
            ('rnbqkbnr/ppp1pppp/3p4/8', 0),
            ('rnbqkbnr/pppp1ppp/4p3/8', 0),
            ('rnbqkbnr/ppppp1pp/5p2/8', 0),
            ('rnbqkbnr/pppppp1p/6p1/8', 0),
            ('rnbqkbnr/ppppppp1/7p/8', 0),
            ('rnbqkbnr/1ppppppp/8/p7', 0),
            ('rnbqkbnr/p1pppppp/8/1p6', 0),
            ('rnbqkbnr/pp1ppppp/8/2p5', 0),
            ('rnbqkbnr/ppp1pppp/8/3p4', 0),
            ('rnbqkbnr/pppp1ppp/8/4p3', 0),
            ('rnbqkbnr/ppppp1pp/8/5p2', 0),
            ('rnbqkbnr/pppppp1p/8/6p1', 0),
            ('rnbqkbnr/ppppppp1/8/7p', 0),
        ]
        next_next_boards = [Board(nnbb+'/'+nnbw.format(i+j if j else j)) for nnbw, i in next_next_boards_white for nnbb, j in next_next_boards_black]
        moves = chess_utils.legal_moves(init_board)
        next_boards = [self.model.simulate_move(init_board, a, self.model.max_player()) for a in moves]
        for board in next_boards:
            moves = chess_utils.legal_moves(board)
            results = [self.model.simulate_move(board, a, self.model.min_player()) for a in moves]
            for result in results:
                assert result in next_next_boards
                next_next_boards.remove(result)
        assert next_next_boards == []
    
    def create_observation_from_board(self, board:Board):
        board_str = board.fen()
        board_strs = board_str.split(' ')
        obs = []
        for castling_right in ['K', 'Q', 'k', 'q']:
            if castling_right in board_strs[2]:
                obs.append(torch.ones(8,8,1))
            else:
                obs.append(torch.zeros(8,8,1))
        
        if board_strs[1] == 'w':
            obs.append(torch.zeros(8,8,1))
        else:
            obs.append(torch.ones(8,8,1))

        clock = torch.zeros(64)
        clock[int(board_strs[4])//2] = 1
        obs.append(clock.reshape(8,8,1).flip(0))

        pad = torch.ones(8,8,1)
        obs.append(pad)

        board_array = []
        for c in board_strs[0]:
            if not c == '/':
                if c.isnumeric():
                    board_array.extend([ord('.')]*int(c))
                else:
                    board_array.extend([ord(c)])
        board_array = torch.tensor(board_array).reshape(8,8,1)
        
        white_pieces = []
        for p in ['P','N','B','R','Q','K']:
            ob = torch.zeros(8,8,1)
            mask = board_array == ord(p)
            ob[mask] = 1
            white_pieces.append(ob)
        black_pieces = []
        for p in ['p','n','b','r','q','k']:
            ob = torch.zeros(8,8,1)
            mask = board_array == ord(p)
            ob[mask] = 1
            black_pieces.append(ob)
        if self.model.curr_player == chess.BLACK:
            black_pieces = [b_p.flip(0) for b_p in black_pieces]
            white_pieces = [w_p.flip(0) for w_p in white_pieces]
            obs.extend(black_pieces)
            obs.extend(white_pieces)
        else:
            obs.extend(white_pieces)
            obs.extend(black_pieces)

        # FEN cannot represent repeat positions, so tests may fail if
        # observation indicates a repeat.
        obs.append(torch.zeros(8,8,1))

        return torch.cat([*obs, self.base_observation['observation'][:,:,7:-13]], -1)

    def __test_sim_obs(self, board:Board):
        self.model.update_curr_player(board)
        result = self.model.simulate_observation(board, self.base_observation)
        legal_moves = chess_utils.legal_moves(board)
        assert not result["observation"].isnan().any()
        assert result["action_mask"].nonzero().flatten().tolist() == sorted(legal_moves)
        observation = self.create_observation_from_board(board)
        for chan in range(111):
            # broken for more granular testing
            assert (result["observation"][:,:,chan] == observation[:,:,chan]).all(), "failed at {}".format(chan)

    def test_simulate_observation_1(self):
        self.__test_sim_obs(Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))

    def test_simulate_observation_2(self):
        board = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
        self.model.update_curr_player(board)
        self.__test_sim_obs(board)