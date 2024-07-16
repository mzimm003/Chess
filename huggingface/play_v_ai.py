from my_chess.scripts import HumanVsBot
from my_chess.learner.models import DeepChessAlphaBeta, DeepChessAlphaBetaConfig
from my_chess.learner.environments import Chess

from huggingface.pushmodels import DCMinMax

import pickle
from pathlib import Path


def main(kwargs=None):
    modelminmax = DCMinMax.from_pretrained("mzimm003/DeepChessReplicationMinMax")
    play = HumanVsBot(
        model=DeepChessAlphaBeta,
        model_config=DeepChessAlphaBetaConfig(
            board_evaluator=best_model_class,
            board_evaluator_config=best_model_config,
            board_evaluator_param_dir=latest_checkpoint,
            max_depth=3,
            move_sort='evaluation'
        ),
        environment=Chess(render_mode="human"),
        extra_model_environment_context=lambda env: {"board":env.board}
    )
    play.run()

if __name__ == "__main__":
    main()

