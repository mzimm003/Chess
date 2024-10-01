from my_chess.scripts.scripts import HumanVsBot
from my_chess.learner.models import DeepChessAlphaBeta, DeepChessAlphaBetaConfig
from my_chess.learner.environments import Chess

import pickle
from pathlib import Path


def main(kwargs=None):
    # best_model_dir = Path("./results/DeepChessEvaluator/ChessEvaluation_d8721_00002_2_batch_size=256,learning_rate_scheduler_config=milestones___ref_ph_33c73b19_gamma_0_75,model_config=r_2024-03-13_10-19-29").resolve()
    best_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ChessEvaluation_9866d_00000_0_learning_rate_scheduler_config=step_size_200_gamma_0_9,model_config=ref_ph_a52f5213,lr=0.1000_2024-07-13_09-18-52").resolve()
    # best_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ModelDistill_1a05b_00000_0_batch_size=256,learning_rate=0.0100,learning_rate_scheduler_config=step_size_1_gamma_0_95,model_config=_2024-03-29_10-45-15").resolve()
    
    best_model_class = None
    best_model_config = None
    with open(best_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        best_model_class = x['model']
        best_model_config = x['model_config']
        # best_model_config.feature_extractor_param_dir = Path(str(best_model_config.feature_extractor_param_dir).replace("/opt","/home/mark"))

    latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'
    play = HumanVsBot(
        model=DeepChessAlphaBeta,
        model_config=DeepChessAlphaBetaConfig(
            board_evaluator=best_model_class,
            board_evaluator_config=best_model_config,
            board_evaluator_param_dir=latest_checkpoint,
            max_depth=8,
            move_sort='evaluation',
            iterate_depths=True
        ),
        environment=Chess(render_mode="human"),
        extra_model_environment_context=lambda env: {"board":env.board}
    )
    play.run()

if __name__ == "__main__":
    main()