from my_chess.scripts.scripts import Train
from my_chess.learner.models import DeepChessEvaluatorConfig
from my_chess.learner.algorithms import ChessEvaluationConfig
from my_chess.learner.datasets import ChessDataWinLossPairs

import ray.air as air
import ray.tune as tune

import torch

from pathlib import Path
import pickle

def main(kwargs=None):
    best_model_dir = Path("/opt/ray/results/ChessFeatureExtractor/AutoEncoder_1557d_00000_0_batch_size=256,model_config=ref_ph_a52f5213,lr=0.0001_2024-07-12_22-30-58").resolve()
    # best_model_dir = Path("./results/ChessFeatureExtractor/AutoEncoder_8e326_00004_4_batch_size=256,model_config=ref_ph_d2c2b490,lr=0.0001_2024-02-27_15-47-24").resolve()
    best_model_class = None
    best_model_config = None
    with open(best_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        best_model_class = x['model']
        best_model_config = x['model_config']

    latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

    train_script = Train(
        # debug=True,
        num_cpus=16,
        num_gpus=0.85,
        # restore='/opt/ray/results/DeepChessEvaluator',
        training_on="ChessDataWinLossPairs",
        algorithm="ChessEvaluation",
        algorithm_config=ChessEvaluationConfig(
            dataset_config=dict(dataset_dir='/opt/datasets/Chess-CCRL-404', static_partners=False),
            # batch_size = tune.grid_search([16384]),
            optimizer=torch.optim.SGD,
            # optimizer=torch.optim.Adam,
            # learning_rate = tune.grid_search([0.1]),
            learning_rate = tune.grid_search([0.1,0.001,0.0001]),
            learning_rate_scheduler=torch.optim.lr_scheduler.StepLR,
            learning_rate_scheduler_config=tune.grid_search([
                dict(step_size=200, gamma=0.9),
                # dict(step_size=1, gamma=0.99),
                # dict(step_size=1, gamma=0.95),
                # dict(step_size=1, gamma=0.9),
                # dict(step_size=1, gamma=0.85),
                # dict(step_size=1, gamma=0.8),
                # dict(step_size=1, gamma=0.75),
                # dict(milestones=range(1,20), gamma=0.75),
                # dict(milestones=range(1,25), gamma=0.75),
                # dict(milestones=range(1,30), gamma=0.75),
                ]),
            data_split=(.8,.1,.1)
        ),
        model="DeepChessEvaluator",
        model_config=tune.grid_search([
            DeepChessEvaluatorConfig(
                feature_extractor=best_model_class,
                feature_extractor_config=best_model_config,
                feature_extractor_param_dir=latest_checkpoint,
                hidden_dims=[512, 256, 128],
                batch_norm=True),
            ]),
        run_config=air.RunConfig(
                            name="DeepChessEvaluator",
                            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5),
                            stop={"training_iteration": 20},
                            ),
    )
    train_script.run()


if __name__ == "__main__":
    main()
