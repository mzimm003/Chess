from ray.tune.trainable import Trainable as Trainabletemp

class Trainable(Trainabletemp):
    def getName(self):
        return self.__class__.__name__

class TrainableConfig:
    def __init__(self) -> None:
        pass

    def asDict(self):
        return self.__dict__
    
    def getName(self):
        return self.__class__.__name__