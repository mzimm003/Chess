import pytest
import shutil
from pathlib import Path
from my_chess.dataset import ChessDataGenerator

@pytest.fixture(scope="session")
def full_pgn_set_dir(tmpdir_factory, pytestconfig):
    db_dir = tmpdir_factory.mktemp("database")
    shutil.copytree(Path("./tests/test_datasets_pgns"), db_dir, dirs_exist_ok=True)
    return db_dir

@pytest.fixture(scope="session")
def hdd_dir(tmpdir_factory, pytestconfig):
    base_dir = Path("/home/user/hdd_datasets")
    db_name = "tmp"
    return base_dir/db_name

@pytest.fixture(scope="session")
def ssd_dir(tmpdir_factory, pytestconfig):
    base_dir = Path("/home/user/ssd_datasets")
    db_name = "tmp"
    return base_dir/db_name