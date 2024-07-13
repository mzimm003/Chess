"""
"""
import pytest
import tempfile
from pathlib import Path
import shutil
import chess

from my_chess.learner.datasets import ChessData
from my_chess.learner.datasets.chess import raw_env
from my_chess.learner.environments import Chess

class TestChessData:
    environment = Chess()
    def test_dataset_creation(self, full_data):
        full_data, _ = full_data
        full_data = Path(full_data)
        assert full_data.exists()
        assert (full_data/"labels").exists()
        assert (full_data/"data").exists()
        assert (full_data/"data"/"1").exists()
        assert (full_data/"data"/"1"/"0.pkl").exists()

    def test_full_observation_record(self, minimal_obs_game_data):
        """
        Exists to remind of the current limitation noted in ChessData, where
        the environment creating observations for the data base overrules the
        PGN file. So, not all positions will necessarily be observed and saved
        for the database.

        minimal_obs_game_data pgn provides a game which is ultimately won, but
        drawn by PettingZoo's Chess about move 53.
        """
        pass

    def test_database_matches_environment(self, short_game_data):
        path, database = short_game_data
        path = Path(path)
        captured_fens = [
            'rnbqk1nr/pp3ppp/8/2bp4/8/8/PPPNNPPP/R1BQKB1R b KQkq - 1 6',
            'rnb1k1nr/pp3ppp/1q6/2bp4/8/8/PPPNNPPP/R1BQKB1R w KQkq - 2 7']
        obs_fens = [
            raw_env.observation_to_fen(database[i][0][None,:])[0] for i in range(len(database))]
        for ob in obs_fens:
            match_one = False
            for cap in captured_fens:
                if ob.split(' ')[0] == cap.split(' ')[0]:
                    match_one = True
            assert match_one

