from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import TensorType

import gymnasium as gym
from torch import nn, Tensor
import numpy as np

from typing import Dict, List

from my_chess.learner.models import ModelRLLIB, ModelRRLIBConfig


class QLearnerConfig(ModelRRLIBConfig):
    def __init__(
            self,
            num_layers:int=2,
            hidden_dim:int=256) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

class QLearner(ModelRLLIB):
    def __init__(
        self,
        obs_space: gym.spaces.Space=None,
        action_space: gym.spaces.Space=None,
        num_outputs: int=None,
        model_config: ModelConfigDict=None,
        name: str=None,
        config:QLearnerConfig = None,
        **kwargs) -> None:
        super().__init__(
            obs_space = obs_space,
            action_space = action_space,
            num_outputs = num_outputs,
            model_config = model_config,
            name = name
            )
        self.config = config
        orig_space = getattr(self.obs_space,"original_space",self.obs_space)
        self.ff = nn.Sequential(
            *[nn.Linear(np.prod(orig_space['observation'].shape), self.config.hidden_dim),
            nn.ReLU()]*(self.config.num_layers-1),
            nn.Linear(self.config.hidden_dim, self.num_outputs)
        )
        # self.ff = FullyConnectedNetwork(
        #     orig_space["observation"],
        #     self.action_space,
        #     self.num_outputs,
        #     {"fcnet_hiddens":self.config.num_layers*[self.config.hidden_dim],
        #      "fcnet_activation":"relu"},
        #     "ff")
        self.probs = nn.Softmax(-1)
        self._features = None
    
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        action_mask = input_dict['obs']['action_mask']
        input_dict["obs"] = input_dict["obs"]["observation"]
        obs = input_dict['obs'].flatten(1)
        obs = obs.to(next(self.parameters()).dtype)
        self._features = self.ff(obs)
        # model_out, state = self.ff(
        #     input_dict=input_dict,
        #     state=state,
        #     seq_lens=seq_lens)
        return self._features, state

    def value_function(self):
        # return self.ff.value_function()
        return self._features.max(-1)[0]