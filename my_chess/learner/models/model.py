from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import gymnasium as gym
from ray.rllib.utils.typing import ModelConfigDict

class ModelRRLIBConfig:
    def __init__(self) -> None:
        pass

    def asDict(self):
        return self.__dict__

class ModelRLLIB(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space = None,
        action_space: gym.spaces.Space = None,
        num_outputs: int = None,
        model_config: ModelConfigDict = None,
        name: str = None):
        super().__init__(
            obs_space = obs_space,
            action_space = action_space,
            num_outputs = num_outputs,
            model_config = model_config,
            name = name
            )
        #super is not calling nn.Module init for unknown reasons
        nn.Module.__init__(self)

    def getModelSpecificParams(self):
        return self.__dict__
    
class ModelConfig:
    def __init__(self) -> None:
        pass

    def asDict(self):
        return self.__dict__

class Model(nn.Module):
    def __init__(
        self):
        super().__init__()

    def getModelSpecificParams(self):
        return self.__dict__