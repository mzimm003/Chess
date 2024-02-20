from my_chess.scripts import Train
from my_chess.learner.algorithms import (
    PPO,
    PPOConfig,
)
from my_chess.learner.callbacks import SelfPlayCallback
from my_chess.learner.models import ToBeNamed
from my_chess.learner.policies import PPOPolicy

import ray.air as air
import ray.tune as tune

def main(kwargs=None):
    train_script = Train(
        # debug=True,
        num_cpus=16,
        num_gpus=0.85,
        training_on="ChessData",
        algorithm="AutoEncoder",
        algorithm_config="AutoEncoderConfig",
        model="DeepChessFE",
        model_config="DeepChessFEConfig",
        run_config=air.RunConfig(
                            name="ChessFeatureExtractor",
                            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=25),
                            # stop={"timesteps_total": 20},
                            stop={"training_iteration": 20},
                            ),
    )
    # (train_script.getAlgConfig()
    #     .multi_agent(
    #             # policy_mapping_fn=policy_mapping_fn,
    #             policy_map_capacity=2,
    #             )
    #     .callbacks(SelfPlayCallback)
    #     .training(
    #         lr_schedule=tune.grid_search([[[1,0.00001],[7000000,0.000001]],
    #                                       [[1,0.00005],[7000000,0.000005],[9000000,0.000001]],
    #                                       [[1,0.000025],[7000000,0.0000025],[9000000,0.000001]]]),
    #         # optimizer={'simple_optimizer':True},
    #         # kl_coeff=.005,
    #     )
    #     .resources(num_gpus=0.85)
    #     .framework(framework='torch')
    #     .rollouts(num_rollout_workers=11, num_envs_per_worker=50)
    # )
    train_script.run()


if __name__ == "__main__":
    main()