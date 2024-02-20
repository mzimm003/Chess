from ray.tune.trainable import Trainable as Trainabletemp
from ray.train.torch import TorchTrainer
from ray.tune.tune import _Config
import torch

import os

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

class Trainable(Trainabletemp):
    def getName(self):
        return self.__class__.__name__

class TrainableConfig(_Config):
    def __init__(
            self,
            num_cpus=None,
            **kwargs) -> None:
        super().__init__()
        self.num_cpus = num_cpus if num_cpus else os.cpu_count()

    def to_dict(self):
        return vars(self)
    
    def getName(self):
        return self.__class__.__name__
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v:
                setattr(self, k, v)