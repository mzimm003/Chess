from my_chess.scripts.scripts import Train
from my_chess.learner.models import DeepChessEvaluatorConfig
from my_chess.learner.algorithms import ModelDistillConfig

import ray.air as air
import ray.tune as tune

import torch

from pathlib import Path
import pickle

def main(kwargs=None):
    teacher_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ChessEvaluation_9866d_00000_0_learning_rate_scheduler_config=step_size_200_gamma_0_9,model_config=ref_ph_a52f5213,lr=0.1000_2024-07-13_09-18-52").resolve()
    teacher_model_class = None
    teacher_model_config = None
    x = None
    with open(teacher_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        teacher_model_class = x['model']
        teacher_model_config = x['model_config']

    teacher_model_param_path = sorted(teacher_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

    fe_model_dir = Path("/opt/ray/results/ChessFeatureExtractor/ModelDistill_19f52_00000_0_batch_size=256,learning_rate=0.0100,learning_rate_scheduler_config=step_size_1_gamma_0_95,model_config=_2024-03-29_01-41-14").resolve()
    fe_model_class = None
    fe_model_config = None
    x = None
    with open(fe_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        fe_model_class = x['model']
        fe_model_config = x['model_config']

    fe_model_param_path = sorted(fe_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

    train_script = Train(
        # debug=True,
        num_cpus=16,
        num_gpus=0.85,
        training_on="ChessDataWinLossPairs",
        algorithm="ModelDistill",
        algorithm_config=ModelDistillConfig(
            dataset_config=dict(dataset_dir='/opt/datasets/Chess-CCRL-404', static_partners=False),
            batch_size = tune.grid_search([256]),
            optimizer=torch.optim.SGD,
            # optimizer=torch.optim.Adam,
            learning_rate = tune.grid_search([0.01]),
            # learning_rate = tune.grid_search([0.0001, 0.00005]),
            learning_rate_scheduler=torch.optim.lr_scheduler.StepLR,
            learning_rate_scheduler_config=tune.grid_search([
                # dict(step_size=200, gamma=0.9),
                # dict(step_size=1, gamma=0.99),
                dict(step_size=1, gamma=0.95),
                # dict(step_size=1, gamma=0.9),
                # dict(step_size=1, gamma=0.85),
                # dict(step_size=1, gamma=0.8),
                # dict(step_size=1, gamma=0.75),
                # dict(milestones=range(1,20), gamma=0.75),
                # dict(milestones=range(1,25), gamma=0.75),
                # dict(milestones=range(1,30), gamma=0.75),
                ]),
            parent_model=teacher_model_class,
            parent_model_config=teacher_model_config,
            parent_model_param_dir=teacher_model_param_path,
        ),
        model="DeepChessEvaluator",
        model_config=tune.grid_search([
            DeepChessEvaluatorConfig(
                feature_extractor=fe_model_class,
                feature_extractor_config=fe_model_config,
                feature_extractor_param_dir=fe_model_param_path,
                hidden_dims=[128, 128],
                batch_norm=False),
            ]),
        run_config=air.RunConfig(
                            name="DeepChessEvaluator",
                            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
                            stop={"training_iteration": 100},
                            ),
    )
    train_script.run()


if __name__ == "__main__":
    main()