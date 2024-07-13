import pytest
import shutil
from pathlib import Path
from my_chess.learner.datasets import ChessData

@pytest.fixture(scope="session")
def full_data(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copytree(Path("./tests/test_datasets_pgns"), db_dir, dirs_exist_ok=True)
    dataset = ChessData(
        dataset_dir=db_dir,
        seed=12345,
        max_games_per_file=10,
        debug=pytestconfig.getoption("debug"))
    return db_dir, dataset

@pytest.fixture(scope="session")
def short_game_data(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copy(Path("./tests/test_datasets_pgns/shortGame.pgn"), db_dir)
    dataset = ChessData(
        dataset_dir=db_dir,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    return db_dir, dataset

@pytest.fixture(scope="session")
def minimal_obs_game_data(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copy(Path("./tests/test_datasets_pgns/minimalObs.pgn"), db_dir)
    dataset = ChessData(
        dataset_dir=db_dir,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    return db_dir, dataset

@pytest.fixture(scope="session")
def all_draws_game_data(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copy(Path("./tests/test_datasets_pgns/allDraws.pgn"), db_dir)
    dataset = ChessData(
        dataset_dir=db_dir,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    return db_dir, dataset