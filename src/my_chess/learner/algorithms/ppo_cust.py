from typing import Optional, Type
from ray.rllib.algorithms.ppo import PPO as PPOtemp, PPOConfig as PPOConfigtemp
from ray.rllib.utils.annotations import override

from my_chess.learner.algorithms import Algorithm, AlgorithmConfig
from my_chess.learner.policies import Policy, PPOPolicy

class PPO(PPOtemp, Algorithm):
    @classmethod
    @override(PPOtemp)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        return PPOPolicy


class PPOConfig(PPOConfigtemp, AlgorithmConfig):
    pass
