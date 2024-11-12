import ray
from ray.util.multiprocessing import Pool
import chess
import chess.engine

ENGINE_PATH = "/home/user/Programming/stockfish/src/stockfish"
EVALUATIONS = 50
DEPTH = 10

def test_one_engine(benchmark):
    @ray.remote
    class ChessAnalyzer:
        def __init__(self):
            self.engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

        def analyse(self, board, limit):
            return self.engine.analyse(board, limit)
        
        def quit(self):
            self.engine.quit()

    def anl(board_eng):
        board = board_eng[0]
        eng = board_eng[1]
        info = ray.get(eng.analyse.remote(board, chess.engine.Limit(time=3)))
        return info['score'].wdl().white().expectation()

    pool = Pool()
    board = chess.Board("rnb1k1nr/pppp1ppp/8/4P3/6q1/4P1P1/PPP1P2P/RN1QKBNR b KQkq - 0 6")
    boards = [(board,ChessAnalyzer.remote())]*10
    def test():
        return pool.map(anl, boards)
    benchmark.pedantic(test, rounds=3, iterations=5)
    r = boards[0][1].quit.remote()
    ray.get(r)

def test_multi_engine(benchmark):
    @ray.remote
    class ChessAnalyzer:
        def __init__(self):
            self.engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

        def analyse(self, board, limit):
            return self.engine.analyse(board, limit)
        
        def quit(self):
            self.engine.quit()

    def anl(board_eng):
        board = board_eng[0]
        eng = board_eng[1]
        info = ray.get(eng.analyse.remote(board, chess.engine.Limit(time=3)))
        return info['score'].wdl().white().expectation()

    pool = Pool()
    board = chess.Board("rnb1k1nr/pppp1ppp/8/4P3/6q1/4P1P1/PPP1P2P/RN1QKBNR b KQkq - 0 6")
    boards = [(board,ChessAnalyzer.remote()) for i in range(10)]
    def test():
        return pool.map(anl, boards)
    benchmark.pedantic(test, rounds=3, iterations=5)
    for b in boards:
        r = b[1].quit.remote()
        ray.get(r)

    

# def test_keep_open(benchmark):

#     def test():
#         engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
#         for i in range(EVALUATIONS):
#             board = chess.Board("rnbq1bnr/pp2k2N/8/2p1p2Q/4N3/8/PPPP1PPP/R1B1KB1R b KQ - 0 1")
#             info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
#         engine.quit()

#     benchmark.pedantic(test, rounds=3, iterations=5)

# def test_keep_open_10(benchmark):

#     def test():
#         engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
#         for i in range(EVALUATIONS):
#             board = chess.Board("rnbq1bnr/pp2k2N/8/2p1p2Q/4N3/8/PPPP1PPP/R1B1KB1R b KQ - 0 1")
#             info = engine.analyse(board, chess.engine.Limit(depth=10))
#         engine.quit()

#     benchmark.pedantic(test, rounds=3, iterations=5)

# def test_keep_open_15(benchmark):

#     def test():
#         engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
#         for i in range(EVALUATIONS):
#             board = chess.Board("rnbq1bnr/pp2k2N/8/2p1p2Q/4N3/8/PPPP1PPP/R1B1KB1R b KQ - 0 1")
#             info = engine.analyse(board, chess.engine.Limit(depth=15))
#         engine.quit()

#     benchmark.pedantic(test, rounds=3, iterations=5)

# def test_keep_open_20(benchmark):

#     def test():
#         engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
#         for i in range(EVALUATIONS):
#             board = chess.Board("rnbq1bnr/pp2k2N/8/2p1p2Q/4N3/8/PPPP1PPP/R1B1KB1R b KQ - 0 1")
#             info = engine.analyse(board, chess.engine.Limit(depth=20))
#         engine.quit()

#     benchmark.pedantic(test, rounds=3, iterations=5)

# def test_open_close(benchmark):

#     def test():
#         for i in range(EVALUATIONS):
#             engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
#             board = chess.Board("rnbq1bnr/pp2k2N/8/2p1p2Q/4N3/8/PPPP1PPP/R1B1KB1R b KQ - 0 1")
#             info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
#             engine.quit()

#     benchmark.pedantic(test, rounds=3, iterations=5)