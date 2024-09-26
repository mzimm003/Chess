from typing import (
    Literal,
    Union
)

from pettingzoo.classic import chess_v6, tictactoe_v3
from pettingzoo import AECEnv
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv as PZE
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

import torch

import gymnasium as gym

class PettingZooEnv(PZE):
    def __init__(self, env:AECEnv):
        MultiAgentEnv().__init__()
        self.env:AECEnv = env
        env.reset()

        self._agent_ids = set(self.env.agents)

        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space

    def observation_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = [next(iter(self._agent_ids))]
            # agent_ids = self._agent_ids
        return {aid: self._observation_space(aid).sample() for aid in agent_ids}

    def observation_space(self, agent_id = None):
        if agent_id is None:
            agent_id = next(iter(self._agent_ids))
        return self._observation_space(agent_id)

    def action_space_sample(self, agent_ids: list = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = [next(iter(self._agent_ids))]
            # agent_ids = self._agent_ids
        return {aid: self._action_space(aid).sample() for aid in agent_ids}

    def action_space(self, agent_id = None):
        if agent_id is None:
            agent_id = next(iter(self._agent_ids))
        return self._action_space(agent_id)

#Necessary to avoid instantiating environment outside of worker, causing shared parameters
#between what should be independent environments.
def env_creator(env, render_mode=None):
    env = env.env(render_mode=render_mode)
    return env

class Environment(PettingZooEnv):
    def __init__(
            self,
            env,
            render_mode=None) -> None:
        super().__init__(env_creator(env, render_mode=render_mode))
        register_env(
            self.__class__.__name__,
            lambda config: PettingZooEnv(env_creator(env, render_mode=render_mode)))
    
    @property
    def render_mode(self) -> str:
        return self.env.render_mode
    
    @property
    def agent_selection(self) -> str:
        return self.env.agent_selection
    
    @property
    def agents(self) -> str:
        return self.env.agents
    
    def render(self):
        return self.env.render()

    def getName(self):
        return self.__class__.__name__
    
    @staticmethod
    def simulate_move(self, board, action, player):
        pass
    
    @staticmethod
    def simulate_observation(self, board, input):
        pass