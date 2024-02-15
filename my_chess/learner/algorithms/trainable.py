from ray.tune.trainable import Trainable as Trainabletemp
from ray.train.torch import TorchTrainer
from ray.tune.tune import _Config

class Trainable(Trainabletemp):
    def getName(self):
        return self.__class__.__name__

class TrainableConfig(_Config):
    def to_dict(self):
        return vars(self)
    
    def getName(self):
        return self.__class__.__name__
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v:
                setattr(self, k, v)