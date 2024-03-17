from my_chess.scripts import HumanVsBot
from my_chess.learner.models import DeepChessAlphaBeta, DeepChessAlphaBetaConfig
from my_chess.learner.environments import Chess

import pickle
from pathlib import Path


def main(kwargs=None):
    best_model_dir = Path("./results/DeepChessEvaluator/ChessEvaluation_0d120_00000_0_batch_size=256,learning_rate_scheduler_config=step_size_1_gamma_0_75,model_config=ref_ph_a52f5213,lr_2024-03-12_19-04-40").resolve()
    best_model_class = None
    best_model_config = None
    with open(best_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        best_model_class = x['model']
        best_model_config = x['model_config']

    latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'
    play = HumanVsBot(
        model=DeepChessAlphaBeta,
        model_config=DeepChessAlphaBetaConfig(
            board_evaluator=best_model_class,
            board_evaluator_config=best_model_config,
            board_evaluator_param_dir=latest_checkpoint,
        ),
        environment=Chess(render_mode="human"),
        extra_model_environment_context=lambda env: {"board":env.board}
    )
    play.run()

if __name__ == "__main__":
    main()