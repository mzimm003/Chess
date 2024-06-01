from my_chess.scripts import Train
from my_chess.learner.models import DeepChessFEConfig
from my_chess.learner.algorithms import ModelDistillConfig

import ray.air as air
import ray.tune as tune

import torch

from pathlib import Path
import pickle

def main(kwargs=None):
    best_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ChessEvaluation_777e5_00000_0_batch_size=256,learning_rate_scheduler_config=step_size_1_gamma_0_95,model_config=ref_ph_a52f5213,lr_2024-03-27_20-58-42").resolve()
    best_model_class = None
    best_model_config = None
    x = None
    with open(best_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        best_model_class = x['model']
        best_model_config = x['model_config']

    latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

    temp_model = best_model_class(input_sample=torch.rand((2,8,8,111)) ,config=best_model_config)
    temp_model.load_state_dict(torch.load(latest_checkpoint))

    teacher_model = x["model_config"].feature_extractor
    teacher_model_config = x["model_config"].feature_extractor_config
    temp_teacher_model_param_path = Path("/tmp/tmp_model.pt")
    torch.save(temp_model.fe.state_dict(), temp_teacher_model_param_path)

    train_script = Train(
        # debug=True,
        num_cpus=16,
        num_gpus=0.85,
        training_on="ChessData",
        algorithm="ModelDistill",
        algorithm_config=ModelDistillConfig(
            dataset_config=dict(dataset_dir='/opt/datasets/Chess-CCRL-404'),
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
            parent_model=teacher_model,
            parent_model_config=teacher_model_config,
            parent_model_param_dir=temp_teacher_model_param_path,
            train_on_teacher_only=True
        ),
        model="DeepChessFE",
        model_config=tune.grid_search([
            DeepChessFEConfig(
                hidden_dims=[128, 128, 128],
                batch_norm=True),
            ]),
        run_config=air.RunConfig(
                            name="ChessFeatureExtractor",
                            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
                            stop={"training_iteration": 100},
                            ),
    )
    train_script.run()


if __name__ == "__main__":
    main()