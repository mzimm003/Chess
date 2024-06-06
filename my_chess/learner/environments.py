"""
Reinforcement Learning Environments


"""

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
    
    # def reset(self, *, seed: Union[int, None] = None, options: Union[MultiAgentDict, None] = None):
    #     ret = super().reset(seed=seed, options=options)
    #     self.convert_to_tensor(ret)
    #     return ret
    
    # def step(self, action):
    #     ret = super().step(action)
    #     self.convert_to_tensor(ret)
    #     return ret
    
    # def convert_to_tensor(self, obj):
    #     if isinstance(obj, dict):
    #         for k in obj:
    #             if isinstance(obj[k], dict):
    #                 self.convert_to_tensor(obj[k])
    #             else:
    #                 obj[k] = torch.tensor(obj[k])
    #     if isinstance(obj, tuple):
    #         for o in obj:
    #             self.convert_to_tensor(o)



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
    
    def getName(self):
        return self.__class__.__name__

class Chess(Environment):
    def __init__(
            self,
            env=None,
            render_mode:Literal[None, "human", "ansi", "rgb_array"] = None) -> None:
        assert render_mode in {None, "human", "ansi", "rgb_array"}
        env = chess_v6 if env is None else env
        super().__init__(env, render_mode=render_mode)

    @staticmethod
    def mirror_board_view(observation):
        '''Based on Petting Zoo Chess environment'''
        # 1. Mirror the board
        if len(observation.shape) == 3:
            # Not batched
            observation = torch.flip(observation, dims=(0,))
        elif len(observation.shape) == 4:
            # Batched
            observation = torch.flip(observation, dims=(1,))
        else:
            raise RuntimeError("Unexpected observation shape when attempting to mirror.")

        # 2. Swap the white 6 channels with the black 6 channels
        static_channels = torch.arange(7)
        white_channels = torch.arange(8)[:,None]*13+7+torch.arange(6)
        black_channels = torch.arange(8)[:,None]*13+7+6+torch.arange(6)
        rep_channels = torch.arange(8)[:,None]*13+7+6+6
        channel_swap = torch.cat((black_channels, white_channels, rep_channels), dim=-1)
        swap_idxs = torch.cat((static_channels, channel_swap.flatten()))
        return observation[..., swap_idxs]
    
class TicTacToe(Environment):
    def __init__(
            self,
            env=None,
            render_mode:Literal[None, "human", "ansi", "rgb_array"] = None) -> None:
        assert render_mode in {None, "human", "ansi", "rgb_array"}
        env = tictactoe_v3 if env is None else env
        super().__init__(env, render_mode=render_mode)
# import copy
# from typing import Dict, Any

# from pettingzoo import AECEnv
# from pettingzoo.classic.tictactoe_v3 import env as tictactoe_v3

# from ray.rllib.env.multi_agent_env import MultiAgentEnv


# class TicTacToe(MultiAgentEnv):
#     """An interface to the PettingZoo MARL environment library.
#     See: https://github.com/Farama-Foundation/PettingZoo
#     Inherits from MultiAgentEnv and exposes a given AEC
#     (actor-environment-cycle) game from the PettingZoo project via the
#     MultiAgentEnv public API.
#     Note that the wrapper has some important limitations:
#     1. All agents have the same action_spaces and observation_spaces.
#        Note: If, within your aec game, agents do not have homogeneous action /
#        observation spaces, apply SuperSuit wrappers
#        to apply padding functionality: https://github.com/Farama-Foundation/
#        SuperSuit#built-in-multi-agent-only-functions
#     2. Environments are positive sum games (-> Agents are expected to cooperate
#        to maximize reward). This isn't a hard restriction, it just that
#        standard algorithms aren't expected to work well in highly competitive
#        games."""

#     def __init__(
#         self,
#         config: Dict[Any, Any] = None,
#         env: AECEnv = None,
#     ):
#         super().__init__()
#         if env is None:
#             self.env = tictactoe_v3()
#         else:
#             self.env = env
#         self.env.reset()
#         # TODO (avnishn): Remove this after making petting zoo env compatible with
#         #  check_env.
#         self._skip_env_checking = True

#         self.config = config
#         # Get first observation space, assuming all agents have equal space
#         self.observation_space = self.env.observation_space(self.env.agents[0])

#         # Get first action space, assuming all agents have equal space
#         self.action_space = self.env.action_space(self.env.agents[0])

#         assert all(
#             self.env.observation_space(agent) == self.observation_space
#             for agent in self.env.agents
#         ), (
#             "Observation spaces for all agents must be identical. Perhaps "
#             "SuperSuit's pad_observations wrapper can help (useage: "
#             "`supersuit.aec_wrappers.pad_observations(env)`"
#         )

#         assert all(
#             self.env.action_space(agent) == self.action_space
#             for agent in self.env.agents
#         ), (
#             "Action spaces for all agents must be identical. Perhaps "
#             "SuperSuit's pad_action_space wrapper can help (usage: "
#             "`supersuit.aec_wrappers.pad_action_space(env)`"
#         )
#         self._agent_ids = set(self.env.agents)

#     def observe(self):
#         return {
#             self.env.agent_selection: self.env.observe(self.env.agent_selection),
#             "state": self.get_state(),
#         }

#     def reset(self, *args, **kwargs):
#         self.env.reset()
#         return (
#             {self.env.agent_selection: self.env.observe(self.env.agent_selection)},
#             {self.env.agent_selection: {}},
#         )

#     def step(self, action):
#         try:
#             self.env.step(action[self.env.agent_selection])
#         except (KeyError, IndexError):
#             self.env.step(action)
#         except AssertionError:
#             # Illegal action
#             print(action)
#             raise AssertionError("Illegal action")

#         obs_d = {}
#         rew_d = {}
#         done_d = {}
#         trunc_d = {}
#         info_d = {}
#         while self.env.agents:
#             obs, rew, done, trunc, info = self.env.last()
#             a = self.env.agent_selection
#             obs_d[a] = obs
#             rew_d[a] = rew
#             done_d[a] = done
#             trunc_d[a] = trunc
#             info_d[a] = info
#             if self.env.terminations[self.env.agent_selection]:
#                 self.env.step(None)
#                 done_d["__all__"] = True
#                 trunc_d["__all__"] = True
#             else:
#                 done_d["__all__"] = False
#                 trunc_d["__all__"] = False
#                 break

#         return obs_d, rew_d, done_d, trunc_d, info_d

#     def close(self):
#         self.env.close()

#     def seed(self, seed=None):
#         self.env.seed(seed)

#     def render(self, mode="human"):
#         return self.env.render(mode)

#     @property
#     def agent_selection(self):
#         return self.env.agent_selection

#     @property
#     def get_sub_environments(self):
#         return self.env.unwrapped

#     def get_state(self):
#         state = copy.deepcopy(self.env)
#         return state

#     def set_state(self, state):
#         self.env = copy.deepcopy(state)
#         return self.env.observe(self.env.agent_selection)