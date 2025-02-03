from my_chess.dataset import ChessData, ChessDataGenerator
import pytest
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader

chunk_sizes = [(0),(.25),(.5),(.75),(1),]
gen_batch_sizes = [(512),]

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_chunking_write_hdd(chunk_size, gen_batch_size, benchmark, hdd_dir):
    with tempfile.TemporaryDirectory(dir=hdd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        benchmark.pedantic(dbg.create_database, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_chunking_read_hdd(chunk_size, gen_batch_size, benchmark, hdd_dir):
    with tempfile.TemporaryDirectory(dir=hdd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        dbg.create_database()
        db = ChessData(Path(tmpdirname))
        dl = DataLoader(
            db,
            batch_size=64,
            num_workers=4,
            collate_fn=db.__class__.collate_wrapper,
            pin_memory=True)
        def iter_db():
            for batch in dl:
                for i in vars(batch).values():
                    i.to(device='cuda')
        benchmark.pedantic(iter_db, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_chunking_write_ssd(chunk_size, gen_batch_size, benchmark, ssd_dir):
    with tempfile.TemporaryDirectory(dir=ssd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        benchmark.pedantic(dbg.create_database, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_chunking_read_ssd(chunk_size, gen_batch_size, benchmark, ssd_dir):
    with tempfile.TemporaryDirectory(dir=ssd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        dbg.create_database()
        db = ChessData(Path(tmpdirname))
        dl = DataLoader(
            db,
            batch_size=64,
            num_workers=4,
            collate_fn=db.__class__.collate_wrapper,
            pin_memory=True)
        def iter_db():
            for batch in dl:
                for i in vars(batch).values():
                    i.to(device='cuda')
        benchmark.pedantic(iter_db, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_write_hdd(chunk_size, gen_batch_size, benchmark, hdd_dir):
    with tempfile.TemporaryDirectory(dir=hdd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        benchmark.pedantic(dbg.create_database, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_read_hdd(chunk_size, gen_batch_size, benchmark, hdd_dir):
    with tempfile.TemporaryDirectory(dir=hdd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        dbg.create_database()
        db = ChessData(Path(tmpdirname))
        dl = DataLoader(
            db,
            batch_size=64,
            num_workers=4,
            collate_fn=db.__class__.collate_wrapper,
            pin_memory=True)
        def iter_db():
            for batch in dl:
                for i in vars(batch).values():
                    i.to(device='cuda')
        benchmark.pedantic(iter_db, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_write_ssd(chunk_size, gen_batch_size, benchmark, ssd_dir):
    with tempfile.TemporaryDirectory(dir=ssd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        benchmark.pedantic(dbg.create_database, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_read_ssd(chunk_size, gen_batch_size, benchmark, ssd_dir):
    with tempfile.TemporaryDirectory(dir=ssd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression='gzip',
            compression_opts=4,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        dbg.create_database()
        db = ChessData(Path(tmpdirname))
        dl = DataLoader(
            db,
            batch_size=64,
            num_workers=4,
            collate_fn=db.__class__.collate_wrapper,
            pin_memory=True)
        def iter_db():
            for batch in dl:
                for i in vars(batch).values():
                    i.to(device='cuda')
        benchmark.pedantic(iter_db, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_write_hdd_no_compression(chunk_size, gen_batch_size, benchmark, hdd_dir):
    with tempfile.TemporaryDirectory(dir=hdd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression=None,
            compression_opts=None,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        benchmark.pedantic(dbg.create_database, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_read_hdd_no_compression(chunk_size, gen_batch_size, benchmark, hdd_dir):
    with tempfile.TemporaryDirectory(dir=hdd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression=None,
            compression_opts=None,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        dbg.create_database()
        db = ChessData(Path(tmpdirname))
        dl = DataLoader(
            db,
            batch_size=64,
            num_workers=4,
            collate_fn=db.__class__.collate_wrapper,
            pin_memory=True)
        def iter_db():
            for batch in dl:
                for i in vars(batch).values():
                    i.to(device='cuda')
        benchmark.pedantic(iter_db, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_write_ssd_no_compression(chunk_size, gen_batch_size, benchmark, ssd_dir):
    with tempfile.TemporaryDirectory(dir=ssd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression=None,
            compression_opts=None,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        benchmark.pedantic(dbg.create_database, rounds=3, iterations=5)

@pytest.mark.parametrize("chunk_size",chunk_sizes)
@pytest.mark.parametrize("gen_batch_size",gen_batch_sizes)
def test_quadrupled_data_chunking_read_ssd_no_compression(chunk_size, gen_batch_size, benchmark, ssd_dir):
    with tempfile.TemporaryDirectory(dir=ssd_dir) as tmpdirname:
        shutil.copytree(Path("./benchmarks/test_datasets_pgns_quadrupled"), tmpdirname, dirs_exist_ok=True)
        dbg = ChessDataGenerator(
            dir=Path(tmpdirname),
            init_size=65536,
            chunk_size=chunk_size,
            max_size=None,
            resize_step=32768,
            gen_batch_size=gen_batch_size,
            compression=None,
            compression_opts=None,
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None)
        dbg.create_database()
        db = ChessData(Path(tmpdirname))
        dl = DataLoader(
            db,
            batch_size=64,
            num_workers=4,
            collate_fn=db.__class__.collate_wrapper,
            pin_memory=True)
        def iter_db():
            for batch in dl:
                for i in vars(batch).values():
                    i.to(device='cuda')
        benchmark.pedantic(iter_db, rounds=3, iterations=5)