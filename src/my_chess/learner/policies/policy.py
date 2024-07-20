
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

class PolicyConfig:
    def __init__(self) -> None:
        pass

    def asDict(self):
        return self.__dict__

class Policy(TorchPolicyV2):
    pass