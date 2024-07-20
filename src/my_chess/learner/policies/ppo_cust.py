from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy as PPOTP
from my_chess.learner.policies import Policy, PolicyConfig

class PPOPolicyConfig(PolicyConfig):
    pass

class PPOPolicy(PPOTP):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)