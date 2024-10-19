import pytest
import shutil
from pathlib import Path
from my_chess.dataset import ChessDataGenerator

@pytest.fixture(scope="session")
def full_data_dir(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copytree(Path("./tests/test_datasets_pgns"), db_dir, dirs_exist_ok=True)
    dbg = ChessDataGenerator(
        db_dir,
        100000,
        gen_batch_size=2,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    dbg.create_database()
    return dbg.database_path

@pytest.fixture(scope="session")
def short_game_data_dir(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copy(Path("./tests/test_datasets_pgns/shortGame.pgn"), db_dir)
    dbg = ChessDataGenerator(
        db_dir,
        100000,
        gen_batch_size=2,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    dbg.create_database()
    return dbg.database_path

@pytest.fixture(scope="session")
def full_short_game_data_dir(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copy(Path("./tests/test_datasets_pgns/shortGame.pgn"), db_dir)
    dbg = ChessDataGenerator(
        db_dir,
        100000,
        gen_batch_size=2,
        exclude_x_initial_moves=0,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    dbg.create_database()
    return dbg.database_path

@pytest.fixture(scope="session")
def minimal_obs_game_data_dir(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copy(Path("./tests/test_datasets_pgns/minimalObs.pgn"), db_dir)
    dbg = ChessDataGenerator(
        db_dir,
        100000,
        gen_batch_size=2,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    dbg.create_database()
    return dbg.database_path

@pytest.fixture(scope="session")
def all_draws_game_data_dir(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copy(Path("./tests/test_datasets_pgns/allDraws.pgn"), db_dir)
    dbg = ChessDataGenerator(
        db_dir,
        100000,
        gen_batch_size=2,
        seed=12345,
        debug=pytestconfig.getoption("debug"))
    dbg.create_database()
    return dbg.database_path