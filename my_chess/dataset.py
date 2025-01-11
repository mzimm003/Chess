from my_chess.environment import Chess

from ml_training_suite.datasets import Dataset, DataHandlerGenerator
from ml_training_suite.training import Trainer
from ml_training_suite.datasets.generator import HDF5DatasetGenerator

from typing import (
    List,
    Union,
    Literal,
    Dict,
    Any,
    Tuple
)
from typing_extensions import override
from functools import partial

from pathlib import Path

import ray
from ray.util.multiprocessing import Pool
from ray.util import ActorPool

import torch
import pandas as pd
import numpy as np
import h5py

from pettingzoo import AECEnv
from pettingzoo.classic.chess import chess_utils
import chess.engine

import enum

import traceback

class PGNGamesItr:
    def __init__(
            self,
            directory:Union[Path, str],
            subset:int=None,
            drop_draws:bool=False,
            seed:int=None) -> None:
        directory = Path(directory)

        self.current = -1
        self.file_data = pd.concat(
            pd.read_csv(pgn, header=None, sep='\n')
            for pgn in directory.glob('*.pgn'))
        
        self.__set_group_index()
        if drop_draws:
            draw_mask = self.file_data.loc[:,0].str.contains('Result "1/2-1/2"')
            draw_idxs = self.file_data.index[draw_mask]
            self.file_data = self.file_data.drop(index=draw_idxs.append(draw_idxs+1))
            self.__set_group_index()
        self.total_games = (self.file_data.index[-1] + 1) // 2
        self.game_idxs = np.arange(self.total_games)

        if subset:
            rng = np.random.default_rng(seed)

            subset_idxs = rng.choice(
                np.arange(self.total_games),
                subset,
                replace=False,
                shuffle=False)
            subset_idxs.sort()
            kept_idxs = np.empty((subset_idxs.size * 2,), dtype=subset_idxs.dtype)
            kept_idxs[0::2] = subset_idxs * 2
            kept_idxs[1::2] = subset_idxs * 2 + 1

            self.file_data = self.file_data.loc[kept_idxs]
            self.__set_group_index()
            self.total_games = (self.file_data.index[-1] + 1) // 2
            self.game_idxs = np.arange(self.total_games)

    def __set_group_index(self):
        file_idxs = self.file_data.loc[:,0].str.contains('\[').diff().cumsum()
        file_idxs.iloc[0] = 0
        self.file_data.set_index(file_idxs, inplace=True)
    
    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self.game_idxs):
            game_idx = self.game_idxs[self.current]
            meta_idx = game_idx * 2
            move_idx = meta_idx + 1
            return (
                game_idx,
                pd.Series(self.file_data.loc[meta_idx,0]),
                pd.Series(self.file_data.loc[move_idx,0]))
        raise StopIteration

class ChessMovesItr:
    def __init__(
            self,
            moves:List[str],
            rng:np.random.Generator,
            exclude_x_initial_moves:int=0,
            save_states_per_game:int=10,
            ) -> None:
        self.moves = moves
        self.rng = rng
        self.save_states_per_game = (10000000
                                     if save_states_per_game is None
                                     else save_states_per_game)
        self.current = -1
        self.save_move_idxs = np.arange(exclude_x_initial_moves, len(moves))
        self.save_move_idxs = self.rng.choice(
            self.save_move_idxs,
            min(self.save_states_per_game, len(self.save_move_idxs)),
            replace=False,
            shuffle=False)
        self.save_move_idxs.sort()
    
    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if (self.current < len(self.moves) 
            and len(self.save_move_idxs) > 0):
            move = self.moves[self.current]
            is_capture = 'x' in move
            should_save = False
            if self.current == self.save_move_idxs[0]:
                should_save = True
                self.save_move_idxs = self.save_move_idxs[1:]
            return (
                move,
                is_capture,
                should_save
                )
        raise StopIteration

class Filters(enum.IntEnum):
    capture=0
    draw=1
    move_1=2
    move_2=3
    move_3=4
    move_4=5
    move_5=6
    move_6=7
    move_7=8
    move_8=9
    move_9=10
    move_10=11

class ChessDataGenerator(HDF5DatasetGenerator):
    """
    Generator of chess databases based on PGN files.

    To create a database, simply provide as many Portable Game Notation (PGN)
    files as desired and create a class instance with the directory containing
    your PGNs as the "dataset_dir". The PGNs will be parsed and the database
    will be created. If a database has already been created in the directory
    given, the database will be quickly loaded by the instance, without
    reprocessing the PGNs.

    Note: Games may not play to completion when creating the database. That is, 
    rules are dictated by `PettingZoo's Chess Environment <https://pettingzoo.farama.org/environments/classic/chess/>`_
    supported by `Python Chess <https://python-chess.readthedocs.io/en/latest/>`_
    which may be different than the rules used in the game provided by the PGN.
    If the chess environment deems the game over before the PGN file, further
    observations cannot be created for the database.
    """
    def __init__(
            self,
            dir: Union[str, Path],
            init_size: int,
            chunk_size: int = 1,
            max_size: int = None,
            resize_step: int = 100000,
            gen_batch_size: int = 1000,
            compression: Literal['gzip', 'lzf', 'szip'] = 'gzip',
            compression_opts: Any = None,
            render_mode:Literal[None, "human", "ansi", "rgb_array"]=None,
            oracle:bool=False,
            oracle_path:Union[str, Path]=None,
            oracle_limit_config:Dict=None,
            exclude_draws:bool=False,
            exclude_x_initial_moves:int=6,
            reset:bool=False,
            states_per_game:int=None,
            subset:int=None,
            seed:int=None,
            debug:bool=False,
            ) -> None:
        """
        Args:
            dir: The directory of where the chess database should be created
            based on the PGN files therein contained.
            init_size: Expected size of dataset.
            chunk_size: Number of records to open at once on access (None is
              all).
            max_size: Largest size the dataset can be.
            gen_batch_size: Number of records to keep in memory before
              saving to disk. (Aids in multiprocessing)
            filters: Categories to include in database to enable filtering.
            render_mode: Whether to render the game boards as the database is 
              created (slow).
            oracle: Whether to include a chess engine evaluation in database.
            oracle_path: Directory path to chess engine.
            exclude_draws: Only save games that are won/lost.
            exclude_x_initial_moves: Only save moves starting with x.
            reset: Whether to overwrite an existing database by reprocessing
              PGNs in the directory.
            states_per_game: The number of board positions for which
              observations will be saved, per game. (None saves all.)
            subset: The number of games to randomly select as some subset of 
              those in the PGN file. To help control the size of the database as
              it will be several orders of magnitude larger than the PGN.
            seed: Seed for reproducibility in randomizing components.
            debug: Flag to aid debugging efforts (slow).
        """
        super().__init__(
            dir=dir,
            init_size=init_size,
            chunk_size=chunk_size,
            max_size=max_size,
            resize_step=resize_step,
            filters=Filters,
            compression=compression,
            compession_opts=compression_opts)
        self.gen_batch_size = gen_batch_size
        self.reset = reset
        self.oracle = oracle
        self.oracle_path = oracle_path
        self.oracle_limit_config = {} if oracle_limit_config is None else oracle_limit_config
        self.oracle_in_need_of_closing: List[chess.engine.SimpleEngine] = []
        if self.oracle:
            assert self.oracle_path, "A path must be provided to oracle chess engine to include oracle evaluations in database."
        self.exclude_draws = exclude_draws
        self.exclude_x_initial_moves = exclude_x_initial_moves
        self.states_per_game = states_per_game
        self.render_mode = render_mode
        self.subset:int = subset
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.debug = debug

    def create_database(self):
        """
        Custom parser for files from http://computerchess.org.uk/ccrl/404/games.html, provided in pgn
        DeepChess suggests draw games are not useful, and will be excluded if apply_deepchess_rules=True.
        """
        ray.init()
        games_text = self.__separate_games_text()
        self.__generate_obs_and_labels(games_text)
        ray.shutdown()
        self.quit_oracles()
        self.repack()

    def __separate_games_text(self):
        def fill_games_text(game_info:Tuple[int, pd.DataFrame, pd.DataFrame]):
                game_count = game_info[0]
                metadata = game_info[1].tolist()
                moves = game_info[2].tolist()
                return game_count, {'metadata':metadata, 'moves':moves}

        game_itr = PGNGamesItr(
            directory=self.dir,
            subset=self.subset,
            drop_draws=self.exclude_draws,
            seed=self.seed)
        games_text = None
        if self.debug:
            games_text = [fill_games_text(g) for g in game_itr]
        else:
            with Pool() as pool:
                games_text = pool.map(fill_games_text, game_itr)
        print('Game File Read')
        return games_text

    def oracle_generator(self, engine:chess.engine.SimpleEngine=None):
        if not self.oracle:
            return None
        if engine is None:
            engine = chess.engine.SimpleEngine.popen_uci(self.oracle_path)
            self.oracle_in_need_of_closing.append(engine)
        return partial(
            engine.analyse,
            limit=chess.engine.Limit(**self.oracle_limit_config))

    def quit_oracles(self):
        for o in self.oracle_in_need_of_closing:
            o.quit()
        self.oracle_in_need_of_closing = []

    def __generate_obs_and_labels(self, games_text:Tuple[int, Dict[str,List[str]]]):
        def process_game(game:Tuple[int, Dict[str,List[str]]], as_dict=True, oracle=None):
            try:
                game_idx = game[0]
                game_text = game[1]
                game:dict = {"game_idx": int(game_idx)}

                for line in game_text['metadata']:
                    metadata = line.strip('\n[] "')
                    key, val = metadata.split(' "')
                    if key == "Result":
                        game[Filters.draw] = val == "1/2-1/2"
                        game["White Wins"] = val != "0-1"
                        game["Black Wins"] = val != "1-0"
                    else:
                        game[key] = val
                moves =[]
                for line in game_text['moves']:
                    line = line.rstrip('\n')
                    for itm in line.split(' '):
                        if not '.' in itm and not itm in set(["1/2-1/2","1-0","0-1"]):
                            moves.append(itm)
                game_obs = self.__transform_game_moves_to_obs(moves, oracle=oracle)
                result = []
                if as_dict:
                    num_records = len(next(iter(game_obs.values())))
                    record_metadata = {}
                    for k, v in game.items():
                        record_metadata[k] = [v]*num_records
                    game_obs.update(record_metadata)
                    result = game_obs
                else:
                    for game_ob in zip(*game_obs.values()):
                        metadata_cpy = game.copy()
                        metadata_cpy.update({k:v for k,v in zip(game_obs.keys(), game_ob)})
                        result.append(metadata_cpy)
                return result
            except Exception as e:
                raise RuntimeError("\nGAME:{}\nERROR:{}\n{}".format(game, e, traceback.format_exc()))

        @ray.remote
        class OracleEngine:
            def __init__(slf) -> None:
                slf.engine = chess.engine.SimpleEngine.popen_uci(self.oracle_path)
            def process_game(slf, game:Tuple[int, Dict[str,List[str]]], as_dict=True):
                return process_game(
                    game,
                    as_dict,
                    oracle=self.oracle_generator(slf.engine))
            def analyse(slf, board, limit):
                return slf.engine.analyse(board, limit)
            def quit(slf):
                slf.engine.quit()

        #Find sample assured to contain data to be captured to init database
        sample = None
        for samp in games_text:
            if len(samp[1]['moves']) > 0:
                dct = samp[1].copy()
                dct['moves'] = dct['moves'][:1]
                sample = (samp[0], dct)
                break
        self.initialize_by_sample(
            process_game(
                sample,
                as_dict=False,
                oracle=self.oracle_generator())[0])
        self.quit_oracles()
        for i in range(len(games_text)//self.gen_batch_size+1):
            batch_start_idx = i*self.gen_batch_size
            batch_end_idx = (i+1)*self.gen_batch_size
            gs = None
            if self.debug:
                oracle = self.oracle_generator()
                gs = [process_game(g, oracle=oracle) for g in games_text[batch_start_idx:batch_end_idx]]
            else:
                if self.oracle:
                    actors = [OracleEngine.remote() for _ in range(int(ray.nodes()[0]['Resources']['CPU']))]
                    pool = ActorPool(actors)
                    gs = pool.map(
                        lambda actor, board: actor.process_game.remote(board),
                        games_text[batch_start_idx:batch_end_idx])
                else:
                    pool = Pool()
                    gs = pool.map(
                        process_game,
                        games_text[batch_start_idx:batch_end_idx])

            for g in gs:
                g = g.copy()
                filts = {}
                for k in Filters:
                    filts[k] = g[k]
                    del g[k]
                self.add_data_by_batch(g,filters=filts)

    def __transform_game_moves_to_obs(self, moves, environment=None, oracle=None):
        """
        DeepChess suggests initial 5 moves and capture moves are not useful positions, and will be excluded if apply_deepchess_rules=True.
        """
        need_close = False
        if environment is None:
            environment = Chess.env(render_mode=self.render_mode)
            environment.reset()
            need_close = True
        observations = []
        action_masks = []
        oracle_insights = []
        captures = []
        move_num = {i:[] for i in range(1,11)}
        agent_perspective = []
        # board_strs = []
        
        chess_moves = ChessMovesItr(
            moves=moves,
            rng=self.rng,
            exclude_x_initial_moves=self.exclude_x_initial_moves,
            save_states_per_game=self.states_per_game
        )
        agent_itr = iter(environment.agent_iter())
        observation, reward, termination, truncation, info = environment.last()
        for i, (move, capture, save_obs) in enumerate(chess_moves):
            try:
                next(agent_itr) #May break early when pettingzoo and PGN rules do not align
                if termination or truncation:
                    action = None
                else:
                    action = self.__convert_algebraic_notation_to_petting_zoo_chess_action(
                        environment=environment,
                        observation=observation,
                        move=move)
                # if store_obs:
                    # board_strs.append(str(environment.board)) #Helpful for debug, but not necessary

                environment.step(action)
                observation, reward, termination, truncation, info = environment.last()

                if save_obs:
                    o_m0 = environment.observe('player_0')
                    o_m1 = environment.observe('player_1')
                    observations.extend([
                        o_m0['observation'],
                        o_m1['observation']])
                    action_masks.extend([
                        o_m0['action_mask'],
                        o_m1['action_mask']])
                    captures.extend([
                        capture,
                        capture])
                    for k in move_num.keys():
                        if k == int(i//2):
                            move_num[k].extend([True,True])
                        else:
                            move_num[k].extend([False,False])
                    agent_perspective.extend([
                        0,
                        1])
                    if not oracle is None:
                        insight = oracle(
                            environment.board)
                        insight = (insight['score']
                                   .wdl(ply=environment.board.ply())
                                   .white()
                                   .expectation())
                        oracle_insights.extend([insight, 1-insight])
            except:
                break
        # game["board_strs"] = board_strs #Helpful for debug, but not necessary
        if need_close:
            environment.close()
        game_results = {
            "observation": observations,
            "action_mask": action_masks,
            Filters.capture: captures,
            Filters.move_1: move_num[1],
            Filters.move_2: move_num[2],
            Filters.move_3: move_num[3],
            Filters.move_4: move_num[4],
            Filters.move_5: move_num[5],
            Filters.move_6: move_num[6],
            Filters.move_7: move_num[7],
            Filters.move_8: move_num[8],
            Filters.move_9: move_num[9],
            Filters.move_10: move_num[10],
            "agent_perspective": agent_perspective}
        if self.oracle:
            game_results["oracle"] = oracle_insights
        return game_results

    def __convert_algebraic_notation_to_petting_zoo_chess_action(self, environment:AECEnv, observation:np.array, move:str):
        """
        From https://pettingzoo.farama.org/environments/classic/chess/:
        Action Space
        From the AlphaZero chess paper:
        [In AlphaChessZero, the] action space is a 8x8x73 dimensional array.
        Each of the 8x8 positions identifies the square from which to “pick up” a piece.
        The first 56 planes encode possible 'queen moves' for any piece:
        a number of squares [1..7] in which the piece will be moved, along one of eight
        relative compass directions {N, NE, E, SE, S, SW, W, NW}.
        
        The next 8 planes encode possible knight moves for that piece.
        The final 9 planes encode possible underpromotions for pawn moves or captures
        in two possible diagonals, to knight, bishop or rook respectively.
        Other pawn moves or captures from the seventh rank are promoted to a queen.
        
        We instead flatten this into 8x8x73 = 4672 discrete action space.
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
    
class ChessDataTrainer(Trainer):
    @override
    def map_data_handler(self, pipeline):
        return DataHandlerGenerator(
            board=ChessDataBatch.CHESS_BOARD_OBS,
            target=ChessDataBatch.TGT,
            pipeline=pipeline if pipeline else [])
    
class ChessDataAutoEncodingTrainer(Trainer):
    @override
    def map_data_handler(self, pipeline):
        return DataHandlerGenerator(
            board=ChessDataBatch.CHESS_BOARD_OBS,
            target=ChessDataBatch.CHESS_BOARD_OBS,
            pipeline=pipeline if pipeline else [])

class ChessDataBatch:
    CHESS_BOARD_OBS="chess_board_observation"
    TGT="tgt"
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.chess_board_observation = torch.stack(transposed_data[0], 0).long()
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.chess_board_observation = self.chess_board_observation.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def chess_data_collate_wrapper(batch):
    return ChessDataBatch(batch)

class ChessData(Dataset):
    """
    A database of chess board observations separated by labels and data.
    """
    UNIQUE_KEY_LABELS = [
        'Event',
        'Date',
        'Round',
        'White',
        'Black',
        ]
    collate_wrapper = chess_data_collate_wrapper
    def __init__(
            self,
            dataset_dir:Union[Path, str],
            seed: int = None,
            filters:List[int]=None,
            debug:bool=False,
            ) -> None:
        """
        Args:
            dataset_dir: The directory of an existing chess database, or where 
              one should be created based on the PGN files therein contained.
            seed: Seed for reproducibility in randomizing components.
            filters: Categories to be excluded from the dataset.
            debug: Flag to aid debugging efforts (slow).
        """
        super().__init__(seed)
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.suffix:
            # assume directory in need of file
            self.dataset_dir = self.dataset_dir/(ChessDataGenerator.ROOTNAME+'.h5')
        self.filters = [] if filters is None else filters
        filters = {k:False for k in self.filters}
        self.debug = debug

        self.valid_indices = None
        with h5py.File(self.dataset_dir, 'r') as f:
            self.valid_indices = np.arange(f.attrs[ChessDataGenerator.CURR_SIZE])
            mask = np.ones_like(self.valid_indices, dtype=bool)
            filter_flags = f[ChessDataGenerator.FILTER_FLAGS][:]
            for bit, value in filters.items():
                if value:
                    mask &= (filter_flags & (1 << bit)).astype(bool)
                else:
                    mask &= ~(filter_flags & (1 << bit)).astype(bool)
            self.valid_indices = self.valid_indices[mask]
            self.clusters = []
            for u_k in ChessData.UNIQUE_KEY_LABELS:
                _, u_k_i = np.unique(f[u_k], return_inverse=True)
                self.clusters.append(u_k_i)
            _, self.clusters = np.unique(self.clusters, return_inverse=True, axis=1)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        ob = None
        result = None
        with h5py.File(self.dataset_dir, 'r') as f:
            ob = f['observation'][real_idx]
            result = np.array([
                f['White Wins'][real_idx],
                f['Black Wins'][real_idx]
                ])
            result = result/result.sum()
        return torch.from_numpy(ob), torch.from_numpy(result)
    
    def __len__(self):
        return len(self.valid_indices)

class ChessDataSmall(ChessData):
    def __init__(
            self,
            dataset_dir: Union[Path, str],
            seed: int = None,
            filters: List[int] = None,
            debug: bool = False,
            size: int = 2048,
            ) -> None:
        super().__init__(dataset_dir, seed, filters, debug)
        rng = np.random.default_rng()
        self.valid_indices = rng.choice(self.valid_indices, size=size, replace=False)

        
class ChessDataWinLossPairsTrainer(Trainer):
    @override
    def map_data_handler(self, pipeline):
        return DataHandlerGenerator(
            board_pair=ChessDataWinLossPairsBatch.CHESS_BOARD_OBS_PAIR,
            target=ChessDataWinLossPairsBatch.TGT,
            pipeline=pipeline if pipeline else [])

class ChessDataWinLossPairsBatch:
    CHESS_BOARD_OBS_PAIR="chess_board_observation_pair"
    TGT="tgt"
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.chess_board_observation_pair = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.chess_board_observation_pair = self.chess_board_observation_pair.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def chess_data_pairs_collate_wrapper(batch):
    return ChessDataWinLossPairsBatch(batch)

class ChessDataWinLossPairs(ChessData):
    collate_wrapper = chess_data_pairs_collate_wrapper
    def __init__(
            self,
            dataset_dir:Union[Path, str],
            seed: int = None,
            filters:List[int]=None,
            debug:bool=False,
            static_partners:bool=True
            ) -> None:
        super().__init__(
            dataset_dir=dataset_dir,
            seed=seed,
            filters=filters,
            debug=debug)
        self.gen = torch.Generator()
        if self.seed:
            self.gen.manual_seed(self.seed)
        self.idx_partners = None
        if static_partners:
            partners = self.get_static_random_idx_partners()
            self.idx_partners = lambda x: partners[x]
        else:
            self.idx_partners = partial(self.get_dynamic_random_idx_partners)
        
    def __getitem__(self, idx):
        pos1 = super().__getitem__(idx)
        pos2 = super().__getitem__(self.idx_partners(idx))
        pos2_ret = pos2[0]
        if (pos1[1]==pos2[1]).all():
            pos2_ret = Chess.mirror_board_view(pos2[0])
        return torch.stack([pos1[0], pos2_ret], dim=-4), pos1[1]

    def get_static_random_idx_partners(self):
        '''
        Will provide a random index to be paired with data loader index Created
        to be consistent with dataset ordering indexes to avoid excessive
        reloading of labels (which exists in batches)
        '''
        return torch.randperm(len(self.valid_indices), generator=self.gen)
    
    def get_dynamic_random_idx_partners(self, idx):
        '''
        Will provide a random index to be paired with data loader index Created
        to be consistent with dataset ordering indexes to avoid excessive
        reloading of labels (which exists in batches)
        '''
        return torch.randint(len(self.valid_indices), size=(1,), generator=self.gen).item()
    

class ChessDataWinLossPairsSmall(ChessDataWinLossPairs):
    def __init__(
            self,
            dataset_dir: Union[Path, str],
            seed: int = None,
            filters: List[int] = None,
            debug: bool = False,
            static_partners: bool = True,
            size: int = 2048,
            ) -> None:
        super().__init__(dataset_dir, seed, filters, debug, static_partners)
        rng = np.random.default_rng()
        self.valid_indices = rng.choice(self.valid_indices, size=size, replace=False)