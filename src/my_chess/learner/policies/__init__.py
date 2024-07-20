"""Policies to control actions based on environment state."""

from .policy import Policy, PolicyConfig
from .ppo_cust import PPOPolicy, PPOPolicyConfig, PPOTP
from .random import RandomPolicy, RandomPolicyConfig


"""Due to the following error: "WARNING policy.py:137 -- Can not figure out a
durable policy name for <class 'my_chess.learner.policies.ppo.PPOPolicy'>.
You are probably trying to checkpoint a custom policy. Raw policy class may cause
problems when the checkpoint needs to be loaded in the future. To fix this, make
sure you add your custom policy in rllib.algorithms.registry.POLICIES." We provide:"""
import ray.rllib.algorithms
ray.rllib.algorithms.__path__.append(*__path__)
from ray.rllib.algorithms.registry import POLICIES

import sys
import inspect

add_policies = {k:v for k,v in inspect.getmembers(sys.modules[__name__], inspect.isclass) if not "Config" in k}
for pol_name, pol in add_policies.items():
    POLICIES[pol_name] = inspect.getmodule(pol).__name__.replace(__name__+'.','')