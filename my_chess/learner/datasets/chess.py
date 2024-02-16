from my_chess.learner.datasets import Dataset
from pettingzoo.classic.chess_v6 import raw_env as r_e
from pettingzoo.utils import wrappers
from pettingzoo.classic.chess import chess_utils
from typing import Tuple, Union, List, Dict
from pathlib import Path
import shutil
import pickle
import json
import numpy as np
import pandas as pd
from ray.util.multiprocessing import Pool
import ray
from functools import partial
from io import TextIOWrapper
import gc
from tqdm import tqdm

from itertools import islice

"""
To prevent memory problems with multiprocessing.
Provided by https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
See https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814 for summary
"""
# --- UTILITY FUNCTIONS ---
def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
    return np.array([ord(c) for c in s], dtype=dtype)

def sequence_to_string(seq: np.ndarray) -> str:
    return ''.join([chr(c) for c in seq])

def pack_sequences(seqs: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(seqs, axis=0)
    offsets = np.cumsum([len(s) for s in seqs])
    return values, offsets

def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    off1 = offsets[index]
    if index > 0:
        off0 = offsets[index - 1]
    elif index == 0:
        off0 = 0
    else:
        raise ValueError(index)
    return values[off0:off1]


"""
Ray multiprocessing does not provide a locking mechanism. The following solution is provided here: https://github.com/ray-project/ray/issues/8017
"""
import posix_ipc

class SystemSemaphore:
    def __init__(self, name=None, limit=1):
        self.name  = f'/{name}' if name else name
        self.limit = limit

    def __enter__(self):
        kwargs = dict(flags=posix_ipc.O_CREAT, mode=384, initial_value=self.limit)
        self.lock = posix_ipc.Semaphore(self.name, **kwargs)
        self.lock.acquire()

    def __exit__(self, _type, value, tb):
        self.lock.release()


"""
chess_v6 env does not provide an initial observation of the board with no pieces moved.
"""
class raw_env(r_e):
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

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class ChessData(Dataset):
    AUTONAME = "complete_generated_dataset"
    def __init__(self, data_dir, seed=None, apply_deepchess_rules=True, render_mode=None, reset=False, max_games_per_file=14000) -> None:
        super().__init__()
        self.reset = reset
        self.apply_deepchess_rules = apply_deepchess_rules
        self.render_mode = render_mode
        self.data_dir = Path(data_dir)
        self.label_dir = self.data_dir/'labels'
        self.obs_dir = self.data_dir/'observations'
        self.max_games_per_file = max_games_per_file
        self.obs_data = {
            "total_label_files":0,
            "total_observations":0,
            "current_file":0,
            "_obs_counts":[0],
            "_cum_obs_counts":[0],
            }
        
        self.__create_structure()
        
        self.obs_data = pd.read_json(self.label_dir/(ChessData.AUTONAME+".json"))
        self.labels = pd.read_json(self.label_dir/"{}-{}.json".format(ChessData.AUTONAME, self.obs_data.loc[0,"current_file"]), orient="records")
    #     self.__apply_numpy()

    # def __apply_numpy(self):
    #     # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    #     pass


        
    def __len__(self):
        return self.obs_data.loc[0,'total_observations']
    
    def __getitem__(self, idx):
        file_idx = self.__necessary_file(idx)
        idx -= self.obs_data.loc[file_idx,"_cum_obs_counts"]
        if self.obs_data.loc[0,"current_file"] != file_idx:
            self.obs_data.loc[0,"current_file"] = file_idx
            self.labels = pd.read_json(self.label_dir/"{}-{}.json".format(ChessData.AUTONAME, self.obs_data.loc[0,"current_file"]), orient="records")

        labels = self.labels.loc[idx]
        ob_path = self.obs_dir/labels['file_name']
        ob = None
        with open(ob_path, 'rb') as f:
            ob = pickle.load(f)
        label = (1
                 if ((labels['Result'] == '1-0' and labels['agent_perspective'] == 'player_0') or
                     (labels['Result'] == '0-1' and labels['agent_perspective'] == 'player_1'))
                 else -1)
        return ob['observation'], label
    
    def __necessary_file(self, idx):
        return np.digitize(idx, self.obs_data.loc[:,"_cum_obs_counts"])-1
    
    def __create_structure(self):
        if (self.reset or
            not (self.label_dir/(ChessData.AUTONAME+".json")).exists()):
            if self.label_dir.exists():
                shutil.rmtree(self.label_dir)
            if self.obs_dir.exists():
                shutil.rmtree(self.obs_dir)
            
            self.label_dir.mkdir(parents=True)
            self.obs_dir.mkdir(parents=True)

            for file in self.data_dir.glob("*.pgn"):
                self.__create_database_from_pgn(file)
            
            self.obs_data["_cum_obs_counts"] = np.cumsum(self.obs_data["_obs_counts"]).tolist()

            with open(self.label_dir/(ChessData.AUTONAME+".json"), 'w') as f:
                json.dump(self.obs_data, f)

    
    def __create_database_from_pgn(self, file:str):
        """
        Custom parser for files from http://computerchess.org.uk/ccrl/404/games.html, provided in pgn
        DeepChess suggests draw games are not useful, and will be excluded if apply_deepchess_rules=True.
        """
        ray.init()
        pool = Pool()
        games_text = self.__separate_games_text(file, pool)
        self.__generate_obs_and_labels(games_text, pool)
        ray.shutdown()

    def __generate_obs_and_labels(self, games_text:Tuple[int, Dict[str,List[str]]], pool:Pool):
        def process_game(game:Tuple[int, Dict[str,List[str]]]):
            game_idx = game[0]
            game_text = game[1]
            game = {'metadata':{}}
            if not self.apply_deepchess_rules or not "1/2-1/2" in game_text['metadata'][6]:
                for line in game_text['metadata']:
                    metadata = line.strip('\n[] "')
                    key, val = metadata.split(' "')
                    game['metadata'][key] = val
                moves =[]
                for line in game_text['moves']:
                    line = line.rstrip('\n')
                    for itm in line.split(' '):
                        if not '.' in itm and not itm in set(["1/2-1/2","1-0","0-1"]):
                            moves.append(itm)
                game["moves"] = moves
                self.__transform_game_moves_to_obs(game)

                metadata = []
                for i, obs in enumerate(game['observations']):
                    file_name = '{}/{}.pkl'.format(game_idx, i)
                    if not Path(self.obs_dir/file_name).parent.exists():
                        Path(self.obs_dir/file_name).parent.mkdir()
                    with open(self.obs_dir/file_name, 'wb') as f:
                        pickle.dump(obs, f)
                    m_d = game['metadata'].copy()
                    m_d['file_name'] = file_name
                    # m_d['board_str'] = game['board_strs'][i] #Helpful for debug, but not necessary
                    m_d['agent_perspective'] = game['agent_perspective'][i]
                    metadata.append(m_d)
                return metadata

        for i in range(self.obs_data["total_label_files"],
                       self.obs_data["total_label_files"]+(len(games_text)//self.max_games_per_file)+1):
            file_start_idx = i*self.max_games_per_file
            file_end_idx = (i+1)*self.max_games_per_file
            gs = pool.map(process_game, games_text[file_start_idx:file_end_idx])
            # gs = list(tqdm(pool.imap(process_game, games_text), total=len(games_text))) #tqdm causes cpu under utilization
            gs = [j for k in gs if k for j in k]
            self.obs_data['total_observations'] += len(gs)
            self.obs_data['_obs_counts'].append(len(gs))
            self.obs_data['total_label_files'] += 1
            games = {'observations':[]}
            for g in gs:
                games['observations'].append(g)
            pd.DataFrame(gs).to_json(self.label_dir/"{}-{}.json".format(ChessData.AUTONAME, i), orient='records')
    
    def __separate_games_text(self, file:str, pool:Pool):
        file_copy = None
        with open(file, 'r') as f:
            file_copy = f.readlines()
        class GamesItr:
            def __init__(self, file_copy:List[str]) -> None:
                self.current = -1
                self.file_data = pd.DataFrame(file_copy)
                file_idxs = (self.file_data=='\n').cumsum().shift(fill_value=0)
                self.file_data.set_index(file_idxs[0], inplace=True)
                self.total_games = file_copy.count('\n')//2
            
            def __iter__(self):
                return self

            def __next__(self):
                self.current += 1
                meta_idx = self.current * 2
                move_idx = meta_idx + 1
                if self.current < self.total_games:
                    return self.current, self.file_data.loc[meta_idx,0], self.file_data.loc[move_idx,0]
                raise StopIteration

        def fill_games_text(game_info:Tuple[int, pd.DataFrame, pd.DataFrame]):
            game_count = game_info[0]
            metadata = game_info[1].tolist()[:-1] #Removes trailing '\n'
            moves = game_info[2].tolist()[:-1] #Removes trailing '\n'
            return game_count, {'metadata':metadata, 'moves':moves}

        game_itr = GamesItr(file_copy=file_copy)
        # games_text = list(tqdm(pool.imap(fill_games_text, game_itr), total=game_itr.total_games)) #tqdm causes cpu under utilization
        games_text = pool.map(fill_games_text, game_itr)
        print('Game File Read')
        return games_text

    def __transform_games_moves_to_obs(self, games, environment=None):
        """
        DeepChess suggests initial 5 moves and capture moves are not useful positions, and will be excluded if apply_deepchess_rules=True.
        """
        if environment is None:
            environment = env()
        for game in games:
            environment.reset()
            self.__transform_game_moves_to_obs(game, environment=environment)
        environment.close()
        return games

    def __transform_game_moves_to_obs(self, game, environment=None):
        """
        DeepChess suggests initial 5 moves and capture moves are not useful positions, and will be excluded if apply_deepchess_rules=True.
        """
        need_close = False
        if environment is None:
            environment = env(render_mode=self.render_mode)
            environment.reset()
            need_close = True
        observations = []
        agent_perspective = []
        # board_strs = []
        move_idx = 0
        store_obs = True
        if self.apply_deepchess_rules:
            store_obs = False

        for agent in environment.agent_iter():
            observation, reward, termination, truncation, info = environment.last()
            if store_obs:
                observations.append(observation)
                agent_perspective.append(agent)

            if termination or truncation:
                action = None
            else:
                action = self.__convert_algebraic_notation_to_petting_zoo_chess_action(
                    environment=environment,
                    observation=observation,
                    move=game["moves"][move_idx])
                
                if self.apply_deepchess_rules:
                    if move_idx > 5 and not 'x' in game["moves"][move_idx]:
                        store_obs = True
                    else:
                        store_obs = False
            # if store_obs:
                # board_strs.append(str(environment.board)) #Helpful for debug, but not necessary


            environment.step(action)

            move_idx += 1
            if move_idx >= len(game["moves"]):
                break
        
        game["observations"] = observations
        game["agent_perspective"] = agent_perspective
        # game["board_strs"] = board_strs #Helpful for debug, but not necessary

        if need_close:
            environment.close()

    def __convert_algebraic_notation_to_petting_zoo_chess_action(self, environment:env, observation:np.array, move:str):
        """
        From https://pettingzoo.farama.org/environments/classic/chess/:
        Action Space
        From the AlphaZero chess paper:
        [In AlphaChessZero, the] action space is a 8x8x73 dimensional array.
        Each of the 8×8 positions identifies the square from which to “pick up” a piece.
        The first 56 planes encode possible ‘queen moves’ for any piece:
        a number of squares [1..7] in which the piece will be moved, along one of eight
        relative compass directions {N, NE, E, SE, S, SW, W, NW}.
        
        The next 8 planes encode possible knight moves for that piece.
        The final 9 planes encode possible underpromotions for pawn moves or captures
        in two possible diagonals, to knight, bishop or rook respectively.
        Other pawn moves or captures from the seventh rank are promoted to a queen.
        
        We instead flatten this into 8×8×73 = 4672 discrete action space.
        You can get back the original (x,y,c) coordinates from the integer action a
        with the following expression: (a // (8*73), (a // 73) % 8, a % (8*73) % 73)
        
        Example: >>> x = 6 >>> y = 0 >>> c = 12 >>> a = x*(873) + y73 + c >>>
          print(a // (873), a % (873) // 73, a % (8*73) % 73)
          6 0 12
        
        Note: the coordinates (6, 0, 12) correspond to column 6, row 0, plane 12.
        In chess notation, this would signify square G1.

        Notes based on experimentation:
        First 56 planes consider compass directions in order {SW, W, NW, S, N, SE, E, NE}.
        The next 8 planes provide knight moves in order {WSW, WNW, SSW, NNW, SSE, NNE, ESE, ENE}.
        Finally, the last 9 planes provide (Dir-Piece) {NW-N, NW-B, NW-R, N-N, N-B, N-R, NE-N, NE-B, NE-R}.

        Observation:
        From https://pettingzoo.farama.org/environments/classic/chess/:
        Channel 7 - 18: One channel for each piece type and player color combination.
        For example, there is a specific channel that represents black knights. An index of
        this channel is set to 1 if a black knight is in the corresponding spot on the game
        board, otherwise, it is set to 0. Similar to LeelaChessZero, en passant possibilities
        are represented by displaying the vulnerable pawn on the 8th row instead of the 5th.

        Notes based on experimentation:
        1st - My pawns
        2nd - My Knights
        3rd - My Bishops
        4th - My Rooks
        5th - My Queen
        6th - My King
        7th - Opponent pawns
        8th - Opponent Knights
        9th - Opponent Bishops
        10th - Opponent Rooks
        11th - Opponent Queen
        12th - Opponent King
        """
        PIECES = {
            "P":0,
            "N":1,
            "B":2,
            "R":3,
            "Q":4,
            "K":5
            }
        COLUMNS = {
            "a":0,
            "b":1,
            "c":2,
            "d":3,
            "e":4,
            "f":5,
            "g":6,
            "h":7
            }
        PROMOTIONS = {
            "N":2,
            "B":3,
            "R":4,
            "Q":5
            }
        
        blacks_move = np.all(observation['observation'][:,:,4])
        obs = np.flip(np.moveaxis(observation['observation'][:,:,7:18], 2, 0), axis=1)
        # obs = obs[6:] if blacks_move else obs[:6]
        obs = np.flip(obs[:6], axis=1) if blacks_move else obs[:6]

        move = move.rstrip('+#')
        piece = None
        piece_context = None
        promotion = None
        if "=" in move:
            promotion = move[-1]
            move = move[:-2]
        
        if move in set(["O-O-O", "O-O"]):
            piece = "K"
            castles = {
                "O-O": "Kg8" if blacks_move else "Kg1",
                "O-O-O": "Kc8" if blacks_move else "Kc1",
            }
            move = castles[move]
        else:
            piece = move[:-2].rstrip('x')
            if not piece:
                piece = "P"
            elif not piece[0].isupper():
                piece_context = piece[0]
                piece = "P"
            elif len(piece) > 1:
                piece_context = piece[-1]
                piece = piece[0]
        
        action_to_take = None
        piece_positions = np.flip(np.array(obs[PIECES[piece]].nonzero()).T, axis=-1)
        for legal_action in chess_utils.legal_moves(environment.board):
            # iterate every legal move available to check whether the move made
            # (supplied in algebraic notation) ends in the same board space
            legal_move = chess_utils.action_to_move(environment.board, legal_action, int(blacks_move))
            legal_move_to = str(legal_move)[2:]
            if move[-2:] == legal_move_to[:2]:
                # Then check that the legal move comes from a piece that made the move
                legal_move_from = chess_utils.move_to_coord(legal_move)
                if (piece_positions==legal_move_from).all(-1).any():
                    # Then make sure there aren't multiples of the same piece that could make the move
                    # If there are, use the piece context that comes with the move made in algebraic notation
                    # Additionally, if move is a pawn promotion, ensure the legal move is the correct promotion
                    appropriate_promotion = True
                    if promotion:
                        appropriate_promotion = PROMOTIONS[promotion] == legal_move.promotion

                    if appropriate_promotion:
                        if not action_to_take is None:
                            if piece_context in COLUMNS:
                                if legal_move_from[0] == COLUMNS[piece_context]:
                                    action_to_take = legal_action
                            else:
                                if legal_move_from[1] == int(piece_context)-1:
                                    action_to_take = legal_action
                        else:
                            action_to_take = legal_action
        
        return action_to_take