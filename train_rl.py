from my_chess.scripts import Train
from my_chess.learner.algorithms import (
    PPO,
    PPOConfig,
)
from my_chess.learner.callbacks import SelfPlayCallback
from my_chess.learner.models import DeepChessRLConfig
from my_chess.learner.policies import PPOPolicy

import ray.air as air
import ray.tune as tune

import pickle
from pathlib import Path
def main(kwargs=None):
    best_model_dir = Path("./results/ChessFeatureExtractor/AutoEncoder_5a829_00000_0_batch_size=256,model_config=ref_ph_a52f5213,lr=0.0001_2024-03-07_00-47-39").resolve()
    best_model_class = None
    best_model_config = None
    with open(best_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        best_model_class = x['model']
        best_model_config = x['model_config']

    latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

    train_script = Train(
        debug=True,
        num_cpus=16,
        num_gpus=0.95,
        training_on="Chess",
        algorithm="PPO",
        algorithm_config="PPOConfig",
        policy="PPOPolicy",
        policy_config="PPOPolicyConfig",
        model="DeepChessRL",
        model_config=DeepChessRLConfig(
            feature_extractor=best_model_class,
            feature_extractor_config=best_model_config,
            feature_extractor_param_dir=latest_checkpoint,
        ),
        run_config=air.RunConfig(
                            name="DeepChessRL",
                            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10)),
    )
    (train_script.getAlgConfig()
        .multi_agent(
                # policy_mapping_fn=policy_mapping_fn,
                policy_map_capacity=2,
                )
        .callbacks(SelfPlayCallback)
        .training(
            lr=tune.grid_search([0.00001]),
            optimizer={'simple_optimizer':True},
            # kl_coeff=.005,
        )
        .resources(num_gpus=0.95)
        .framework(framework='torch')
        .rollouts(num_rollout_workers=1, num_envs_per_worker=5)
    )
    train_script.run()

if __name__ == "__main__":
    main()