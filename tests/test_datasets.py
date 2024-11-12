"""
"""
import pytest
import tempfile
from pathlib import Path
import shutil
import chess

from my_chess.dataset import ChessData, ChessDataGenerator
from my_chess.environment import Chess, raw_env

class TestChessData:
    def test_dataset_creation(self, full_data_dir):
        assert full_data_dir.exists()

    def test_dataset_repack(self, full_data_dir):
        assert False

    def test_dataset_filter1(self):
        """
        Test ability to filter out positions representing a piece capture.
        """
        #TODO
        assert False

    def test_dataset_filter3(self):
        """
        Test ability to filter out draw games.
        """
        #TODO
        assert False

    def test_full_observation_record(self, minimal_obs_game_data_dir):
        """
        Exists to remind of the current limitation noted in ChessData, where
        the environment creating observations for the data base overrules the
        PGN file. So, not all positions will necessarily be observed and saved
        for the database.

        minimal_obs_game_data pgn provides a game which is ultimately won, but
        drawn by PettingZoo's Chess about move 53.
        """
        pass

    def test_database_matches_environment1(self, short_game_data_dir):
        path = short_game_data_dir
        database = ChessData(path)
        captured_fens = [
            'rnbqkbnr/pp3ppp/4p3/2pP4/3P4/8/PPPN1PPP/R1BQKBNR b KQkq - 0 4',
            'rnbqkbnr/pp3ppp/8/2pp4/3P4/8/PPPN1PPP/R1BQKBNR w KQkq - 0 5',
            'rnbqkbnr/pp3ppp/8/2Pp4/8/8/PPPN1PPP/R1BQKBNR b KQkq - 0 5',
            'rnbqk1nr/pp3ppp/8/2bp4/8/8/PPPN1PPP/R1BQKBNR w KQkq - 0 6',
            'rnbqk1nr/pp3ppp/8/2bp4/8/8/PPPNNPPP/R1BQKB1R b KQkq - 1 6',
            'rnb1k1nr/pp3ppp/1q6/2bp4/8/8/PPPNNPPP/R1BQKB1R w KQkq - 2 7',
            ]
        obs_fens = [
            raw_env.observation_to_fen(database[i][0][None,:])[0] for i in range(len(database))]
        for ob in obs_fens:
            match_one = False
            for cap in captured_fens:
                if ob.split(' ')[0] == cap.split(' ')[0]:
                    match_one = True
            assert match_one

    def test_database_matches_environment2(self, full_short_game_data_dir):
        path = full_short_game_data_dir
        database = ChessData(path)
        captured_fens = [
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
            'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
            'rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2',
            'rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3',
            'rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq - 1 3',
            'rnbqkbnr/pp3ppp/4p3/2pp4/3PP3/8/PPPN1PPP/R1BQKBNR w KQkq c6 0 4',
            'rnbqkbnr/pp3ppp/4p3/2pP4/3P4/8/PPPN1PPP/R1BQKBNR b KQkq - 0 4',
            'rnbqkbnr/pp3ppp/8/2pp4/3P4/8/PPPN1PPP/R1BQKBNR w KQkq - 0 5',
            'rnbqkbnr/pp3ppp/8/2Pp4/8/8/PPPN1PPP/R1BQKBNR b KQkq - 0 5',
            'rnbqk1nr/pp3ppp/8/2bp4/8/8/PPPN1PPP/R1BQKBNR w KQkq - 0 6',
            'rnbqk1nr/pp3ppp/8/2bp4/8/8/PPPNNPPP/R1BQKB1R b KQkq - 1 6',
            'rnb1k1nr/pp3ppp/1q6/2bp4/8/8/PPPNNPPP/R1BQKB1R w KQkq - 2 7',
            ]
        obs_fens = [
            raw_env.observation_to_fen(database[i][0][None,:])[0] for i in range(len(database))]
        for ob in obs_fens:
            match_one = False
            for cap in captured_fens:
                if ob.split(' ')[0] == cap.split(' ')[0]:
                    match_one = True
            assert match_one

