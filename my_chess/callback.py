from typing import List, Union
from ml_training_suite.callbacks.premades import SupervisedTrainingCallback
from ml_training_suite.callbacks.metrics import Metric

import numpy as np

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy

class CustomSupervisedTrainingCallback(SupervisedTrainingCallback):
    def __init__(
            self,
            target_label_probability_dim: Union[int, bool] = False,
            model_label_probability_dim: Union[int, bool] = False
            ) -> None:
        super().__init__(
            target_label_probability_dim=target_label_probability_dim,
            model_label_probability_dim=model_label_probability_dim)
        self.TRAINING_METRICS=[
            Metric.Loss,
            Metric.LearningRate]
        self.VALIDATION_METRICS=[
            Metric.Loss,
            Metric.Accuracy,
            Metric.Precision,
            Metric.Recall,
            Metric.LearningRate,
            ]
        

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
        # gradFldr = Path('./grads')
        # if not gradFldr.exists():
        #     gradFldr.mkdir()
        main_policy = algorithm.get_policy("default_policy")
        algorithm.remove_policy('default_policy')

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            # agent_id = [0|1] -> policy depends on episode ID
            # This way, we make sure that both policies sometimes play agent0
            # (start player) and sometimes agent1 (player to move 2nd).
            return "main" if episode.custom_metrics['curr_ep'] % 2 == int(agent_id.split('_')[-1]) % 2 else "main_v0"
        
        algorithm.add_policy(policy_id='main_v0', policy=main_policy)
        self.opponent['current'] = 'main_v0'
        if hasattr(algorithm.config, '_enable_learner_api') and algorithm.config._enable_learner_api:
            algorithm.add_policy(
                policy_id='main',
                policy=main_policy,
                module_spec=SingleAgentRLModuleSpec.from_module(main_policy.model),
                policies_to_train=['main'],
                policy_mapping_fn=policy_mapping_fn
            )
        else:
            algorithm.add_policy(
                policy_id='main',
                policy=main_policy,
                policies_to_train=['main'],
                policy_mapping_fn=policy_mapping_fn
            )
        

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.

        # grads = [par.grad.cpu().numpy() for par in algorithm.get_policy('main').model.parameters()]
        # torch.save(grads,'./grads/{}.pt'.format(algorithm.training_iteration))
        # with open('test.txt', 'a') as f:
        #     json.dump(grads, f)
        if "wins_mean" in result['custom_metrics']:
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
                            if episode.custom_metrics['curr_ep'] % 2 == int(agent_id.split('_')[-1]) % 2
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
        else:
            print(f"Iter={algorithm.iteration} No games yet finished, will keep learning ...")

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
