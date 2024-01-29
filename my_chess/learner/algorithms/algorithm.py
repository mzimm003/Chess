from ray.rllib.algorithms import Algorithm as Algorithmtemp, AlgorithmConfig as AlgorithmConfigtemp

class Algorithm(Algorithmtemp):
    def getName(self):
        return self.__class__.__name__

class AlgorithmConfig(AlgorithmConfigtemp):
    def getName(self):
        return self.__class__.__name__