"""RL Environment"""

from typing import (
    Literal
)

from pettingzoo.classic import chess_v5
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv

class Environment:
    def __init__(
            self,
            env) -> None:
        self.env = env
        register_env(
            self.__class__.__name__,
            lambda config: PettingZooEnv(env))
    
    def getName(self):
        return self.__class__.__name__

class Chess(Environment):
    def __init__(
            self,
            env=None,
            render_mode:Literal[None, "human", "ansi", "rgb_array"] = None) -> None:
        assert render_mode in {None, "human", "ansi", "rgb_array"}
        env = chess_v5.env(render_mode) if env is None else env
        super().__init__(env)