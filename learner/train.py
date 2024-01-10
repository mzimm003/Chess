from pettingzoo.classic import chess_v5

import os
import argparse
import json
from pathlib import Path
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.random_agent import RandomAgent
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import (PolicySpec,Policy)
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.rollout_worker import RolloutWorker

import numpy as np
import torch

from random_policy import RandomPolicy

from learner import ToBeNamed

# from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
# from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec

class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.latest_pot_opponent = 0
        self.oppPolicies = []
        self.oppPolicyDir = './opponents/'
        self.oppPolicyToBeRemoved = None
        self.opponent = {
            'toBeRemoved':None,
            'current':None
        }
        # self.next_opp = 0
        self.curr_ep = 0

    def set_next_opp(self, next_opp):
        self.next_opp = next_opp

    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        **kwargs,
    ) -> None:
        gradFldr = Path('./grads')
        if not gradFldr.exists():
            gradFldr.mkdir()
        main_policy = algorithm.get_policy("default_policy")
        algorithm.remove_policy('default_policy')

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # agent_id = [0|1] -> policy depends on episode ID
            # This way, we make sure that both policies sometimes play agent0
            # (start player) and sometimes agent1 (player to move 2nd).
            return "main" if episode.custom_metrics['curr_ep'] % 2 == int(agent_id.split('_')[-1]) else "main_v0"
        
        algorithm.add_policy(policy_id='main_v0', policy_cls=type(main_policy))
        self.opponent['current'] = 'main_v0'
        if hasattr(algorithm.config, '_enable_learner_api') and algorithm.config._enable_learner_api:
            algorithm.add_policy(
                policy_id='main',
                policy_cls=type(main_policy),
                config=main_policy.config,
                module_spec=SingleAgentRLModuleSpec.from_module(main_policy.model),
                policies_to_train=['main'],
                policy_mapping_fn=policy_mapping_fn
            )
        else:
            algorithm.add_policy(
                policy_id='main',
                policy_cls=type(main_policy),
                config=main_policy.config,
                policies_to_train=['main'],
                policy_mapping_fn=policy_mapping_fn
            )
        

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.

        grads = [par.grad.cpu().numpy() for par in algorithm.get_policy('main').model.parameters()]
        torch.save(grads,'./grads/{}.pt'.format(algorithm.training_iteration))
        # with open('test.txt', 'a') as f:
        #     json.dump(grads, f)

        win_rate = result['custom_metrics']['wins_mean']
        result["win_rate"] = win_rate
        print(f"Iter={algorithm.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > 0.95:
        # if win_rate > 0.05:
            # Remove current opponent policy (to save RAM). Remote workers should finish episode, then update, hence delayed remove.
            if not self.opponent['toBeRemoved'] is None and not self.opponent['toBeRemoved'] == self.opponent['current']:
                algorithm.remove_policy(policy_id=self.opponent['toBeRemoved'], )
            self.opponent['toBeRemoved'] = self.opponent['current']

            # Introduce new potential opponent to be selected in next training bout. Represents current state of trained model.
            self.latest_pot_opponent += 1
            new_pol_id = f"main_v{self.latest_pot_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")
            self.oppPolicies.append(new_pol_id)

            main_policy = algorithm.get_policy("main")
            main_state = main_policy.get_state()
            new_policy = Policy.from_state(main_state)
            new_policy.export_checkpoint(self.oppPolicyDir+new_pol_id)

            # Choose new opponent policy from those checkpoints saved to disk

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").
            self.opponent['current'] = np.random.choice(self.oppPolicies)
            if not self.opponent['toBeRemoved'] == self.opponent['current']:
                opp_choice_id = self.opponent['current']
                def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                    # agent_id = [0|1] -> policy depends on episode ID
                    # This way, we make sure that both policies sometimes play
                    # (start player) and sometimes agent1 (player to move 2nd).
                    return (
                        "main"
                        if episode.custom_metrics['curr_ep'] % 2 == int(agent_id.split('_')[-1])
                        else opp_choice_id
                    )
                
                opp_policy = Policy.from_checkpoint(self.oppPolicyDir+opp_choice_id)
                algorithm.add_policy(
                    policy_id=opp_choice_id,
                    policy=opp_policy,
                    policy_mapping_fn=policy_mapping_fn
                )

            algorithm.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.latest_pot_opponent + 2

    def on_episode_start(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index: int,
        **kwargs
    ):
        episode.custom_metrics['curr_ep'] = self.curr_ep

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index: int,
        **kwargs
    ):
        self.curr_ep += 1
        rew_main = [y for x,y in episode.agent_rewards.items() if 'main' in x]
        if rew_main and rew_main[0] == -1:
            episode.custom_metrics['wins'] = 0
        else:
            episode.custom_metrics['wins'] = 1

def main(args):
    if args.debug:
        ray.init(num_cpus=1, num_gpus=0, local_mode=True)
    else:
        ray.init(num_cpus=os.cpu_count(), num_gpus=1, local_mode=False)
    
    tuner = None
    if args.restore:
        tuner = tune.Tuner.restore(args.restore, "PPO", resume_errored=True)
    else:
        tuner = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(
                            local_dir="./results",
                            name="test",
                            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=25)),
            param_space=PPOConfig()
                            .environment("Chess", env_config={"render_mode":"human"})
                            .multi_agent(
                                # policy_mapping_fn=policy_mapping_fn,
                                policy_map_capacity=2,
                            )
                            .callbacks(SelfPlayCallback)
                            .training(
                                lr=tune.grid_search([0.0001]),
                                model={
                                    'custom_model': 'my_torch_model',
                                    'custom_model_config': {}
                                    # 'fcnet_hiddens':tune.grid_search([[1280,]]),#,[1280,1280]]),#,[2560,],[2560,2560]]),
                                    # 'dim':8,
                                    # 'conv_filters':[[20,[1,1],1]],
                                    # 'use_attention':True,
                                    # # The number of transformer units within GTrXL.
                                    # # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
                                    # # b) a position-wise MLP.
                                    # "attention_num_transformer_units": 1,
                                    # # The input and output size of each transformer unit.
                                    # "attention_dim": 64,
                                    # # The number of attention heads within the MultiHeadAttention units.
                                    # "attention_num_heads": 1,
                                    # # The dim of a single head (within the MultiHeadAttention units).
                                    # "attention_head_dim": 32,
                                    # # The memory sizes for inference and training.
                                    # "attention_memory_inference": 50,
                                    # "attention_memory_training": 50,
                                    # # The output dim of the position-wise MLP.
                                    # "attention_position_wise_mlp_dim": 32,
                                    # # The initial bias values for the 2 GRU gates within a transformer unit.
                                    # "attention_init_gru_gate_bias": 2.0,
                                    # # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
                                    # "attention_use_n_prev_actions": 0,
                                    # # Whether to feed r_{t-n:t-1} to GTrXL.
                                    # "attention_use_n_prev_rewards": 0,
                                    },
                                optimizer={'simple_optimizer':True},
                            )
                            # .resources(num_gpus=0.25)
                            .resources(num_gpus=0.85)
                            .framework(framework='torch')
                            .rollouts(num_rollout_workers=2, num_envs_per_worker=50),
        )

    tuner.fit()
    ray.shutdown()
    print("Done")


if __name__ == "__main__":        
    def env_creator(args):
        env = chess_v5.env()
        return env

    register_env("Chess",lambda config: PettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("my_torch_model", ToBeNamed)

    parser = argparse.ArgumentParser(description='Run Learner.')
    parser.add_argument('--debug', help='Set to debug.', type=bool)
    parser.add_argument('--restore', help='Tuner checkpoint path from which to continue.', type=str)

    args = parser.parse_args()
    main(args)
