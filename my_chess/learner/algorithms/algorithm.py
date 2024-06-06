"""
Base class to support the training algorithm of a reinforcement learning agent.
"""

from ray.rllib.algorithms import Algorithm as Algorithmtemp, AlgorithmConfig as AlgorithmConfigtemp
import torch

class Algorithm(Algorithmtemp):
    def getName(self):
        return self.__class__.__name__

class AlgorithmConfig(AlgorithmConfigtemp):
    def getName(self):
        return self.__class__.__name__