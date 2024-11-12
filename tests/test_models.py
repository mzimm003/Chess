# """
# """
# import pytest

# import torch
# from pettingzoo.classic.chess import chess_utils
# from chess import Board
# import chess

# from ml_training_suite.models import Model, ModelConfig

# from my_chess.models import DeepChessAlphaBeta, DeepChessAlphaBetaConfig
# from my_chess.environment import Chess
# from my_chess.models.deepchess import NextPositionsGenerator

# class TestDeepChessAlphaBeta:
#     class DummyEvaluatorConfig(ModelConfig):
#         def __init__(self, depth=10):
#             super().__init__()
#             self.depth = depth

#     class DummyEvaluator(Model):
#         def __init__(self, input_sample, config):
#             super().__init__()
#             self.ff = torch.nn.Sequential(*[torch.nn.Identity()]*config.depth)
        
#         def forward(self, input):
#             x = self.ff(input)
#             return torch.ones(x.shape[0],2,device=x.device) * .5

#     environment = Chess()
#     input_sample, _ = environment.reset()
#     input_sample = next(iter(input_sample.values()))
#     model = DeepChessAlphaBeta(
#         input_sample=input_sample,
#         config=DeepChessAlphaBetaConfig(
#             board_evaluator=DummyEvaluator,
#             board_evaluator_config=DummyEvaluatorConfig(),
#             max_depth=2
#         )
#     )
#     base_observation = {"observation": torch.cat([
#         torch.ones(8,8,7)*torch.nan,
#         torch.arange(8).repeat_interleave(13).repeat(8,8,1)], dim=-1)}
    
#     @classmethod
#     def gen_chess_env(cls):
#         environment = Chess()
#         input_sample, _ = environment.reset()
#         return environment

#     def test_env_max_player_1(self):
#         environment = TestDeepChessAlphaBeta.gen_chess_env()
#         self.model.update_curr_player(environment.env.board)
#         assert self.model.max_player() == 0

#     def test_env_max_player_2(self):
#         environment = TestDeepChessAlphaBeta.gen_chess_env()
#         environment.step({'player_0':77})
#         self.model.update_curr_player(environment.env.board)
#         assert self.model.max_player() == 1

#     def test_env_min_player_1(self):
#         environment = TestDeepChessAlphaBeta.gen_chess_env()
#         self.model.update_curr_player(environment.env.board)
#         assert self.model.min_player() == 1

#     def test_env_min_player_2(self):
#         environment = TestDeepChessAlphaBeta.gen_chess_env()
#         environment.step({'player_0':77})
#         self.model.update_curr_player(environment.env.board)
#         assert self.model.min_player() == 0

#     def test_simulate_move_1(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         self.model.update_curr_player(board)
#         next_boards = [
#             Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/2P5/PP1PPPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPP1P/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/8/7P/PPPPPPP1/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq - 0 1'),
#             Board('rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq - 0 1'),
#         ]
#         moves = chess_utils.legal_moves(board)
#         results = [Chess.simulate_move(board, a, self.model.max_player()) for a in moves]
#         for result in results:
#             assert result in next_boards
#             next_boards.remove(result)
#         assert next_boards == []

#     def test_simulate_move_2(self):
#         init_board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         self.model.update_curr_player(init_board)
#         next_next_boards_white = [
#             ('8/7N/PPPPPPPP/RNBQKB1R w KQkq - {} 2', 1),
#             ('8/5N2/PPPPPPPP/RNBQKB1R w KQkq - {} 2', 1),
#             ('8/N7/PPPPPPPP/R1BQKBNR w KQkq - {} 2', 1),
#             ('8/2N5/PPPPPPPP/R1BQKBNR w KQkq - {} 2', 1),
#             ('8/P7/1PPPPPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('8/1P6/P1PPPPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('8/2P5/PP1PPPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('8/3P4/PPP1PPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('8/4P3/PPPP1PPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('8/5P2/PPPPP1PP/RNBQKBNR w KQkq - {} 2', 0),
#             ('8/6P1/PPPPPP1P/RNBQKBNR w KQkq - {} 2', 0),
#             ('8/7P/PPPPPPP1/RNBQKBNR w KQkq - {} 2', 0),
#             ('P7/8/1PPPPPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('1P6/8/P1PPPPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('2P5/8/PP1PPPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('3P4/8/PPP1PPPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('4P3/8/PPPP1PPP/RNBQKBNR w KQkq - {} 2', 0),
#             ('5P2/8/PPPPP1PP/RNBQKBNR w KQkq - {} 2', 0),
#             ('6P1/8/PPPPPP1P/RNBQKBNR w KQkq - {} 2', 0),
#             ('7P/8/PPPPPPP1/RNBQKBNR w KQkq - {} 2', 0),
#         ]
#         next_next_boards_black = [
#             ('rnbqkb1r/pppppppp/7n/8', 1),
#             ('rnbqkb1r/pppppppp/5n2/8', 1),
#             ('r1bqkbnr/pppppppp/n7/8', 1),
#             ('r1bqkbnr/pppppppp/2n5/8', 1),
#             ('rnbqkbnr/1ppppppp/p7/8', 0),
#             ('rnbqkbnr/p1pppppp/1p6/8', 0),
#             ('rnbqkbnr/pp1ppppp/2p5/8', 0),
#             ('rnbqkbnr/ppp1pppp/3p4/8', 0),
#             ('rnbqkbnr/pppp1ppp/4p3/8', 0),
#             ('rnbqkbnr/ppppp1pp/5p2/8', 0),
#             ('rnbqkbnr/pppppp1p/6p1/8', 0),
#             ('rnbqkbnr/ppppppp1/7p/8', 0),
#             ('rnbqkbnr/1ppppppp/8/p7', 0),
#             ('rnbqkbnr/p1pppppp/8/1p6', 0),
#             ('rnbqkbnr/pp1ppppp/8/2p5', 0),
#             ('rnbqkbnr/ppp1pppp/8/3p4', 0),
#             ('rnbqkbnr/pppp1ppp/8/4p3', 0),
#             ('rnbqkbnr/ppppp1pp/8/5p2', 0),
#             ('rnbqkbnr/pppppp1p/8/6p1', 0),
#             ('rnbqkbnr/ppppppp1/8/7p', 0),
#         ]
#         next_next_boards = [Board(nnbb+'/'+nnbw.format(i+j if j else j)) for nnbw, i in next_next_boards_white for nnbb, j in next_next_boards_black]
#         moves = chess_utils.legal_moves(init_board)
#         next_boards = [Chess.simulate_move(init_board, a, self.model.max_player()) for a in moves]
#         for board in next_boards:
#             moves = chess_utils.legal_moves(board)
#             results = [Chess.simulate_move(board, a, self.model.min_player()) for a in moves]
#             for result in results:
#                 assert result in next_next_boards
#                 next_next_boards.remove(result)
#         assert next_next_boards == []
    
#     def __create_observation_from_board(self, board:Board):
#         board_str = board.fen()
#         board_strs = board_str.split(' ')
#         obs = []
#         for castling_right in ['K', 'Q', 'k', 'q']:
#             if castling_right in board_strs[2]:
#                 obs.append(torch.ones(8,8,1))
#             else:
#                 obs.append(torch.zeros(8,8,1))
        
#         if board_strs[1] == 'w':
#             obs.append(torch.zeros(8,8,1))
#         else:
#             obs.append(torch.ones(8,8,1))

#         clock = torch.zeros(64)
#         clock[int(board_strs[4])//2] = 1
#         clock = clock.reshape(8,8,1)
#         if self.model.curr_player == chess.BLACK:
#             clock = clock.flip(0)
#         obs.append(clock)

#         pad = torch.ones(8,8,1)
#         obs.append(pad)

#         board_array = []
#         for c in board_strs[0]:
#             if not c == '/':
#                 if c.isnumeric():
#                     board_array.extend([ord('.')]*int(c))
#                 else:
#                     board_array.extend([ord(c)])
#         board_array = torch.tensor(board_array).reshape(8,8,1)
        
#         white_pieces = []
#         for p in ['P','N','B','R','Q','K']:
#             ob = torch.zeros(8,8,1)
#             mask = board_array == ord(p)
#             ob[mask] = 1
#             white_pieces.append(ob)
#         black_pieces = []
#         for p in ['p','n','b','r','q','k']:
#             ob = torch.zeros(8,8,1)
#             mask = board_array == ord(p)
#             ob[mask] = 1
#             black_pieces.append(ob)
#         if self.model.curr_player != chess.BLACK:
#             black_pieces = [b_p.flip(0) for b_p in black_pieces]
#             white_pieces = [w_p.flip(0) for w_p in white_pieces]
#             obs.extend(black_pieces)
#             obs.extend(white_pieces)
#         else:
#             obs.extend(white_pieces)
#             obs.extend(black_pieces)

#         # FEN cannot represent repeat positions, so tests may fail if
#         # observation indicates a repeat.
#         obs.append(torch.zeros(8,8,1))

#         a_m = torch.zeros(4672)
#         a_m[chess_utils.legal_moves(board)] = 1

#         return {
#             "observation":torch.cat([*obs, self.base_observation['observation'][:,:,7:-13]], -1),
#             "action_mask":a_m
#         }

#     def __compare_obs(self, result_obs, target_obs):
#         assert not result_obs["observation"].isnan().any()
#         assert (result_obs["action_mask"] == target_obs["action_mask"]).all()
#         assert result_obs["observation"].shape[-1] == target_obs["observation"].shape[-1]
#         for chan in range(target_obs["observation"].shape[-1]):
#             # broken for more granular testing
#             assert (result_obs["observation"][:,:,chan] == target_obs["observation"][:,:,chan]).all(), "failed at {}".format(chan)


#     def __test_sim_obs(self, board:Board):
#         self.model.update_curr_player(board)
#         result = Chess.simulate_observation(board, self.base_observation, self.model.curr_player)
#         observation = self.__create_observation_from_board(board)
#         self.__compare_obs(result, observation)

#     def test_simulate_observation_0(self):
#         """
#         To confirm test observations align with petting zoo chess environment
#         observations.
#         """
#         # First with white's play and black's observation
#         environment = TestDeepChessAlphaBeta.gen_chess_env()
#         observation, *_ = environment.step({"player_0":77})
#         self.model.update_curr_player(environment.env.board)
#         observation = observation["player_1"]
#         for r in observation:
#             observation[r] = torch.from_numpy(observation[r].copy())
#         result = self.__create_observation_from_board(environment.env.board)
#         observation["observation"] = observation["observation"][:,:,:20]
#         result["observation"] = result["observation"][:,:,:20]
#         self.__compare_obs(result, observation)

#         # Then black's play and white's observation
#         observation, *_ = environment.step({"player_1":661})
#         self.model.update_curr_player(environment.env.board)
#         observation = observation["player_0"]
#         for r in observation:
#             observation[r] = torch.from_numpy(observation[r].copy())
#         result = self.__create_observation_from_board(environment.env.board)
#         observation["observation"] = observation["observation"][:,:,:20]
#         result["observation"] = result["observation"][:,:,:20]
#         self.__compare_obs(result, observation)


#     def test_simulate_observation_1(self):
#         self.__test_sim_obs(Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))

#     def test_simulate_observation_2(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
#         self.model.update_curr_player(board)
#         self.__test_sim_obs(board)

#     def test_simulate_observation_3(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
#         self.__test_sim_obs(board)

#     def test_max_player_0(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         self.model.update_curr_player(board)
#         assert self.model.max_player() == 0

#     def test_max_player_1(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
#         self.model.update_curr_player(board)
#         assert self.model.max_player() == 1

#     def test_min_player_0(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         self.model.update_curr_player(board)
#         assert self.model.min_player() == 1

#     def test_min_player_1(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
#         self.model.update_curr_player(board)
#         assert self.model.min_player() == 0
    
#     def test_compare_boards(self):
#         board1 = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         board2 = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
#         ob1 = self.__create_observation_from_board(board1)
#         ob2 = self.__create_observation_from_board(board2)
#         assert (self.model.compare_boards(ob1["observation"], ob2["observation"]) == 0).all()

#     def test_utility_1(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         self.model.update_curr_player(board)

#         end_board = Board('r2qkbnr/pbpppQpp/1pn5/8/2B5/4P3/PPPP1PPP/RNB1K1NR b KQkq - 0 4')

#         assert self.model.utility(end_board) == 1

#     def test_utility_2(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
#         self.model.update_curr_player(board)

#         end_board = Board('r2qkbnr/pbpppQpp/1pn5/8/2B5/4P3/PPPP1PPP/RNB1K1NR b KQkq - 0 4')

#         assert self.model.utility(end_board) == -1

#     def test_utility_3(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         self.model.update_curr_player(board)

#         end_board = Board('8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 99 50')

#         assert self.model.utility(end_board) == 0

#     def test_utility_4(self):
#         board = Board('rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1')
#         self.model.update_curr_player(board)

#         end_board = Board('8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 99 50')

#         assert self.model.utility(end_board) == 0

#     def test_min_max_observations1(self, short_game_data_dir):
#         environment = TestDeepChessAlphaBeta.gen_chess_env()
#         board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
#         self.model.update_curr_player(board)
#         game = [Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'),
#             Board('rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'),
#             Board('rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2'),
#             Board('rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3'),
#             Board('rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq - 1 3'),
#             Board('rnbqkbnr/pp3ppp/4p3/2pp4/3PP3/8/PPPN1PPP/R1BQKBNR w KQkq c6 0 4'),
#             Board('rnbqkbnr/pp3ppp/4p3/2pP4/3P4/8/PPPN1PPP/R1BQKBNR b KQkq - 0 4'),
#             Board('rnbqkbnr/pp3ppp/8/2pp4/3P4/8/PPPN1PPP/R1BQKBNR w KQkq - 0 5'),
#             Board('rnbqkbnr/pp3ppp/8/2Pp4/8/8/PPPN1PPP/R1BQKBNR b KQkq - 0 5'),
#             Board('rnbqk1nr/pp3ppp/8/2bp4/8/8/PPPN1PPP/R1BQKBNR w KQkq - 0 6'),
#             Board('rnbqk1nr/pp3ppp/8/2bp4/8/8/PPPNNPPP/R1BQKB1R b KQkq - 1 6'),
#             Board('rnb1k1nr/pp3ppp/1q6/2bp4/8/8/PPPNNPPP/R1BQKB1R w KQkq - 2 7')]
        
#         npg = NextPositionsGenerator('none')
#         lat_observ = {k:torch.zeros_like(v) for k,v in self.base_observation.items()}
#         for i, state in enumerate(game):
#             nps = npg.generate_next_positions(
#                 board=board,
#                 latest_observation=lat_observ,
#                 turn_player=self.model.max_player() if i%2==0 else self.model.min_player(),
#                 perspective_player=self.model.min_player() if i%2==0 else self.model.max_player())
#             for a, b, obs in nps:
#                 if b == state:
#                     board = b
#                     lat_observ = obs
#                     curr_observation, *_ = environment.step({environment.env.agent_selection:a})
#                     curr_observation = {k:torch.tensor(v.copy()) for k,v in list(curr_observation.values())[0].items()}
#                     next_observation, *_ = environment.env.last()
#                     next_observation = {k:torch.tensor(v.copy()) for k,v in next_observation.items()}
#                     for k in lat_observ:
#                         assert (lat_observ[k] == curr_observation[k]).all()
#                     for k in lat_observ:
#                         assert (lat_observ[k] == next_observation[k]).all()