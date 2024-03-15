from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import TensorType

import gymnasium as gym
import torch
from torch import nn, Tensor
import numpy as np

from typing import Dict, List, Union, Type
from pathlib import Path

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
    
class DeepChessEvaluatorConfig(ModelConfig):
    ACTIVATIONS = {
        'relu':nn.ReLU
    }
    def __init__(
            self,
            feature_extractor:Type[Model]=None,
            feature_extractor_config:ModelConfig=None,
            feature_extractor_param_dir:Union[str, Path]=None,
            hidden_dims:Union[int, List[int]]=[512, 252, 128],
            activations:Union[str, List[str]]='relu'
            ) -> None:
        super().__init__()
        self.feature_extractor:Type[Model] = feature_extractor
        self.feature_extractor_config = feature_extractor_config
        self.feature_extractor_param_dir = feature_extractor_param_dir
        self.hidden_dims = hidden_dims
        self.activations = ([DeepChessFEConfig.ACTIVATIONS[a] for a in activations]
                            if isinstance(activations, list)
                            else [DeepChessFEConfig.ACTIVATIONS[activations] for i in range(len(hidden_dims))])
        
    def __str__(self) -> str:
        return "Shape<{}>".format(self.hidden_dims)

class DeepChessEvaluator(Model):
    def __init__(
        self,
        input_sample,
        config:DeepChessEvaluatorConfig = None) -> None:
        super().__init__()
        self.config = config
        self.fe = self.config.feature_extractor(input_sample=input_sample, config=self.config.feature_extractor_config)
        if self.config.feature_extractor_param_dir:
            self.fe.load_state_dict(torch.load(self.config.feature_extractor_param_dir))

        fe_params = next(iter(self.fe.parameters()))
        input_sample = input_sample.to(dtype=fe_params.dtype, device=fe_params.device)
        sample_fe_output = self.fe(input_sample)
        ff = []
        for i, lyr_dim in enumerate(self.config.hidden_dims):
            if i == 0:
                ff.append(nn.Linear(sample_fe_output.shape[1:].numel()*2, lyr_dim))
            else:
                ff.append(nn.Linear(self.config.hidden_dims[i-1], lyr_dim))
            # if i < len(self.config.hidden_dims)-1:
            ff.append(self.config.activations[i]())
        ff.append(nn.Linear(self.config.hidden_dims[-1], 2))
        self.ff = nn.Sequential(*ff)
        self.probs = nn.Softmax(-1)
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        feat_pos_1 = self.fe(input[:,0])
        feat_pos_2 = self.fe(input[:,1])
        logits = self.ff(torch.cat((feat_pos_1, feat_pos_2),-1))
        probs = self.probs(logits)
        return probs