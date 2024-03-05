from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import TensorType

import gymnasium as gym
from torch import nn, Tensor
import numpy as np

from typing import Dict, List, Union

from my_chess.learner.models import ModelRLLIB, ModelRRLIBConfig, Model, ModelConfig


class DeepChessFEConfig(ModelConfig):
    ACTIVATIONS = {
        'relu':nn.ReLU
    }
    def __init__(
            self,
            hidden_dims:Union[int, List[int]]=[4096, 1024, 256, 128],
            activations:Union[str, List[str]]='relu'
            ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = ([DeepChessFEConfig.ACTIVATIONS[a] for a in activations]
                            if isinstance(activations, list)
                            else [DeepChessFEConfig.ACTIVATIONS[activations] for i in range(len(hidden_dims)-1)])
        
    def __str__(self) -> str:
        return "Shape<{}>".format(self.hidden_dims)

class DeepChessFE(Model):
    def __init__(
        self,
        input_sample,
        config:DeepChessFEConfig = None) -> None:
        super().__init__()
        self.config = config
        ff = []
        for i, lyr_dim in enumerate(self.config.hidden_dims):
            if i == 0:
                ff.append(nn.Linear(input_sample.shape[1:].numel(), lyr_dim))
            else:
                ff.append(nn.Linear(self.config.hidden_dims[i-1], lyr_dim))
            if i < len(self.config.hidden_dims)-1:
                ff.append(self.config.activations[i]())
        self.flatten = nn.Flatten(-3)
        self.ff = nn.Sequential(*ff)
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        
        flt = self.flatten(input)
        logits = self.ff(flt)
        return logits