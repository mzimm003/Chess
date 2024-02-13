from typing import Optional, Type
from ray.rllib.algorithms.ppo import PPO as PPOtemp
from ray.rllib.utils.annotations import override
from torch.utils.data import DataLoader

from my_chess.learner.algorithms import Trainable, TrainableConfig
from my_chess.learner.policies import Policy, PPOPolicy
from my_chess.learner.datasets import Dataset

class AutoEncoderConfig(TrainableConfig):
    def __init__(
            self,
            dataset:Dataset=None,
            batch_size:int=1,
            shuffle:bool=True) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

class AutoEncoder(Trainable):
    def setup(self, config:AutoEncoderConfig):
        self.dataset = config.dataset
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=config.shuffle)

