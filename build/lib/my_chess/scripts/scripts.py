import argparse
import string
import inspect
import abc
import os
import sys
import math
import time
from typing import (
    Type,
    Dict,
    List,
    Union,
    Literal,
    Callable
)
from pathlib import Path
from functools import partial

import ray
from ray import tune, air
from ray.tune.registry import register_trainable

import torch
import numpy as np

import pygame

import my_chess.learner.environments
from my_chess.learner.environments import (
    Environment,
    Chess,
)
ENVIRONMENTS = {k:v for k,v in inspect.getmembers(my_chess.learner.environments, inspect.isclass)}

import my_chess.learner.datasets
from my_chess.learner.datasets import (
    Dataset,
)
DATASETS = {k:v for k,v in inspect.getmembers(my_chess.learner.datasets, inspect.isclass)}

import my_chess.learner.algorithms
from my_chess.learner.algorithms import (
    Algorithm,
    AlgorithmConfig,
    Trainable,
    TrainableConfig,
)
ALGORITHMS = {k:v for k,v in inspect.getmembers(my_chess.learner.algorithms, inspect.isclass) if not "Config" in k}
ALGORITHMCONFIGS = {k:v for k,v in inspect.getmembers(my_chess.learner.algorithms, inspect.isclass) if "Config" in k}

import my_chess.learner.policies
from my_chess.learner.policies import (
    Policy,
    PolicyConfig,
)
POLICIES = {k:v for k,v in inspect.getmembers(my_chess.learner.policies, inspect.isclass) if not "Config" in k}
POLICYCONFIGS = {k:v for k,v in inspect.getmembers(my_chess.learner.policies, inspect.isclass) if "Config" in k}

import my_chess.learner.models
from my_chess.learner.models import (
    Model,
    ModelConfig,
    ModelRLLIB,
    ModelRRLIBConfig
)
MODELS = {k:v for k,v in inspect.getmembers(my_chess.learner.models, inspect.isclass) if not "Config" in k}
MODELCONFIGS = {k:v for k,v in inspect.getmembers(my_chess.learner.models, inspect.isclass) if "Config" in k}

from my_chess.learner.callbacks.callbacks import DefaultCallbacks

import chess
from pettingzoo.classic.chess import chess_utils

class ArgumentCollector:
    DOCSTRARGSHEAD = "Args:"
    ARGHELPSEP = ": "
    def __init__(self, script, parser=None) -> None:
        super().__init__()
        self.parser = argparse.ArgumentParser() if parser is None else parser
        self.script:Script = script
        self.abstract_method_args = { # UPDATE IF additional methods requiring command line arg input are introduced
            # self.script.run:set([]),
            # self.script.setup:set([]),
            self.script.__init__:set([])
        }
        self.args = {}
        self.short_flag_taken = {k:False for k in string.ascii_lowercase}
        self.collectArgs()
        self.subparsers = {}

    def collectArgs(self):
        '''
        Automatically generate necessary arguments for some script for each abstract method.
        '''
        for method in self.abstract_method_args.keys():
            self.__addArgs(method)

    def __addArgs(self, method):
        params = inspect.signature(method).parameters
        param_help = self.__parseDocStr(inspect.getdoc(method))
        for n, p in params.items():
            if not n in ["kwargs", "self"]:
                self.__addArg(p, method, param_help[p.name])

    def __addArg(self, param, method, help):
        flags = []
        if not self.short_flag_taken[param.name[:1]]:
            flags.append('-{}'.format(param.name[:1]))
            self.short_flag_taken[param.name[:1]] = True
        flags.append('--{}'.format(param.name.replace('_','-')))
        if param._annotation is bool:
            self.parser.add_argument(
                *flags,
                action='store_true',
                help=help)
        else:
            self.parser.add_argument(
                *flags,
                action='store',
                default=param.default,
                type=param._annotation,
                help=help)
        self.abstract_method_args[method].add(param.name)

    def __parseDocStr(self, docstr:str):
        args_start_idx = docstr.index(ArgumentCollector.DOCSTRARGSHEAD)
        args = docstr[args_start_idx:]
        help = {}
        for line in args.split('\n'):
            if ArgumentCollector.ARGHELPSEP in line:
                k, h = line.split(ArgumentCollector.ARGHELPSEP)
                help[k.strip()] = h.strip()
        return help

    def getParser(self):
        return self.parser
    
    def addSubParser(self, title, name):
        if not title in self.subparsers:
            self.subparsers[title] = self.parser.add_subparsers(title=title, dest=title)
        return self.subparsers[title].add_parser(name)

    def parseArgs(self, args=None, namespace=None):
        self.args = vars(self.parser.parse_args(args, namespace))
    
    def getArgs(self, method):
        if method in self.args:
            return self.args[method]
        elif method in self.abstract_method_args:
            return {k:self.args[k] for k in self.abstract_method_args[method]}
        else:
            return {}
    
    def setAbstractMethodArgs(self, ama):
        self.abstract_method_args = ama
    
    def getAbstractMethodArgs(self):
        return self.abstract_method_args

class Script(abc.ABC):
    def __init__(self, args=None, namespace=None, parser=None) -> None:
        super().__init__()
        self.argCol = ArgumentCollector(self, parser=parser)
        self.args = args
        self.namespace = namespace

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def parseArgs(self):
        self.argCol.parseArgs(self.args, self.namespace)

    def getArgumentCollector(self):
        return self.argCol
    
    def complete_run(self):
        """
        Runs argument parser before running script.

        Useful for allowing files to run scripts dynamically from the terminal.
        """
        self.parseArgs()
        self.run()

class ScriptChooser(Script):
    def __init__(self, **kwargs) -> None:
        """
        Args:
        """
        super().__init__(**kwargs)
        self.scripts:Dict[str,Script] = {script.__name__:script for script in [
            clss for name, clss in inspect.getmembers(sys.modules[__name__], inspect.isclass) if not name in ['ArgumentCollector','Script','ScriptChooser']
        ]}
        self.script_title = "Scripts"
        self.selected_script = None
        self.selected_script_args = None
    
    def __selectScript(self):
        self.selected_script = self.scripts[self.argCol.getArgs(self.script_title)]
        self.argCol.setAbstractMethodArgs(self.selected_script.getArgumentCollector().getAbstractMethodArgs())
        self.selected_script.update(**self.argCol.getArgs(self.selected_script.__init__))

    def run(self):
        """
        Args:
            args: To pass args within py files.
            namespace: To pass kwargs within py files.
        """
        self.__selectScript()
        self.selected_script.run(**self.argCol.getArgs(self.selected_script.run))

    def parseArgs(self):
        for n, scr in self.scripts.items():
            temp_parse = self.argCol.addSubParser(self.script_title, n)
            self.scripts[n] = scr(parser=temp_parse)
        super().parseArgs()

class Test(Script):
    def __init__(
            self,
            checkpoint:List[Union[str,Path]]=None,
            environment:Union[Environment, str]=None,
            **kwargs) -> None:
        """
        Args:
            checkpoints: directory location of policy to visualize
            environment: Name of environment (if any) to test in.
        """
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.environment = ENVIRONMENTS[environment]() if isinstance(environment, str) else environment
        self.policies = None
        if self.checkpoint:
            if not isinstance(self.checkpoint, list):
                self.checkpoint = [self.checkpoint]
            self.policies = [Policy.from_checkpoint(chkpt) for chkpt in self.checkpoint]
    
    def run(self):
        """
        Args:
        """
        pol1 = pol2 = self.policies[0]

        for i in range(50):
            self.environment.env.reset()
            epRew = 0
            done = False
            j = 0
            p1_turn = True
            while not done:
                observation, reward, termination, truncation, info = self.environment.env.last()
                done = termination or truncation
                # for k,v in observation.items():
                #     if isinstance(v, np.ndarray):
                #         observation[k] = v.tolist()
                # observation = {"obs_flat":observation["obs_flat"]["observation"]}
                if not done:
                    act = None
                    if not p1_turn:
                        act = pol2.compute_single_action(obs=observation)[0]
                    else:
                        act = pol1.compute_single_action(obs=observation)[0]
                    p1_turn = not p1_turn

                    self.environment.env.step(act)
                time.sleep(.3)

class HumanVsBot(Test):
    def __init__(
            self,
            checkpoint:List[Union[str,Path]]=None,
            model:Union[Type[Model],List[Type[Model]]]=None,
            model_config:Union[ModelConfig,List[ModelConfig]]=None,
            environment:Union[Environment, str]=None,
            extra_model_environment_context:Callable=None,
            **kwargs) -> None:
        """
        Args:
            checkpoint: directory location of policy to visualize
            model: extra means of creating a policy. Will overwrite checkpoint input.
            model_config: configuration parameters of model, must be supplied if model is.
            environment: Name of environment (if any) to test in.
            extra_model_environment_context: function which operates on environment to provide additional input to model.
        """
        super().__init__(checkpoint=checkpoint, environment=environment, **kwargs)
        self.model = model
        self.model_config = model_config
        input_sample, _ = self.environment.reset()
        input_sample = next(iter(input_sample.values()))
        self.extra_model_environment_context = extra_model_environment_context if extra_model_environment_context else lambda x: {}
        if self.model:
            if not isinstance(self.model, list):
                self.model = [self.model]
                self.model_config = [self.model_config]
            self.policies = [
                m(input_sample=input_sample, config=m_c)
                if isinstance(m, Type) else
                m
                for m, m_c in zip(self.model, self.model_config)]
            for p in self.policies:
                p.eval()
        
        assert len(self.policies) == len(self.environment.agents) - 1
        pols = [partial(self.get_ai_input, model=mod) for mod in self.policies]
        pols.append(self.get_human_input)
        agent_assignment = np.random.choice(len(pols), len(self.environment.agents), replace=False)
        self.action_map = {}
        self.human_player = None
        for i, j in enumerate(agent_assignment):
            self.action_map[self.environment.agents[i]] = pols[j]
            if pols[j] == self.get_human_input:
                self.human_player = i

    def pos_to_square(self, x, y):
        window_size = pygame.display.get_surface().get_size()
        square_width = window_size[0] // 8
        square_height = window_size[1] // 8
        return x // square_width, (window_size[1] - y) // square_height

    def square_num(self, x, y):
        return y * 8 + x

    def square_to_coord(self, x, y):
        return ''.join((
            chr(ord('a') + x),
            str(y + 1)
            ))
    
    def squares_to_move(self, f, t):
        return self.square_to_coord(*f) + self.square_to_coord(*t)

    def get_human_input(self, observation, **kwargs):
        #RANDOM PLAYER
        # options = torch.arange(observation['action_mask'].size)[observation['action_mask'].astype(bool)]
        # choice = torch.randint(options.numel(), (1,))
        # return options[choice].item()
        action = None
        from_coord = None
        legal_actions = chess_utils.legal_moves(self.environment.env.board)
        legal_moves = [str(chess_utils.action_to_move(self.environment.env.board, x, self.human_player)) for x in legal_actions]
        legally_moved = False
        while not legally_moved:
            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    from_coord = self.pos_to_square(x, y)
                if event.type == pygame.MOUSEBUTTONUP:
                    x, y = event.pos
                    to_coord = self.pos_to_square(x, y)
                    if to_coord == from_coord:
                        # clicked to pick piece
                        attempted_move = False
                        while not attempted_move:
                            ev = pygame.event.get()
                            for event in ev:
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    x, y = event.pos
                                    to_coord = self.pos_to_square(x, y)
                                if event.type == pygame.MOUSEBUTTONUP:
                                    x, y = event.pos
                                    if self.pos_to_square(x, y) == to_coord:
                                        attempted_move = True
                                        move = self.squares_to_move(from_coord, to_coord)
                                        if not move in legal_moves:
                                            move = move + 'q' #hack to incorporate promoting to queen
                                        if move in legal_moves:
                                            action = legal_actions[legal_moves.index(move)]
                                            legally_moved = True
                    else:
                        # dragged piece
                        move = self.squares_to_move(from_coord, to_coord)
                        if not move in legal_moves:
                            move = move + 'q' #hack to incorporate promoting to queen
                        if move in legal_moves:
                            action = legal_actions[legal_moves.index(move)]
                            legally_moved = True
        return action

    def get_ai_input(self, observation, env, model:Union[Policy, Model]):
        act = None
        if isinstance(model, Policy):
            act, _ = model.compute_single_action(obs=observation)
        else:
            act = model(input=observation, **self.extra_model_environment_context(env))
        return act
    
    def run(self):
        """
        Args:
        """
        done = False
        while not done:
            observation, reward, termination, truncation, info = self.environment.env.last()
            done = termination or truncation
            if not done:
                act = self.action_map[self.environment.env.agent_selection](observation=observation, env=self.environment.env)
                self.environment.env.step(act)

class Train(Script):
    def __init__(
            self,
            setup_file:str=None,
            debug:bool=False,
            restore:str='',
            num_cpus:int=1,
            num_gpus:float=0.,
            local_mode:bool=False,
            training_on:Union[Environment, str]=None,
            algorithm:Union[Trainable, Algorithm, str]=None,
            algorithm_config:Union[TrainableConfig, AlgorithmConfig, str]=None,
            policy:Union[Policy, str]=None,
            policy_config:Union[PolicyConfig, str]=None,
            model:Union[ModelRLLIB, str]=None,
            model_config:Union[ModelRRLIBConfig, str]=None,
            run_config:Union[air.RunConfig, str]=None,
            callback:DefaultCallbacks=None,
            framework:Literal["tf","torch"] = "torch",
            **kwargs) -> None:
        """
        Create a training script for a machine learning model.

        Args:
            setup_file: Path to training parameters.
            debug: Boolean determining wether to enable debug mode.
            restore: Tuner checkpoint path from which to continue.
            num_cpus: Number of cpus to allocate to session.
            num_gpus: Number of gpus to allocate to session (including fractional amounts).
            local_mode: Boolean limiting session to be run locally.
            training_on: Name of training set, either an environment or dataset.
            algorithm: Name of training algorithm.
            algorithm_config: Name of training algorithm configuration (named defaults will be used).
            policy: Name of action policy.
            policy_config: Name of action policy configuration (named defaults will be used).
            model: Name of model being trained.
            model_config: Name of training algorithm configuration (named defaults will be used).
            run_config: Name of run configuration to be used.
            callback: Name of in training callback to be used.
            framework: Name of framework to be used (e.g. torch or tf)
        """
        super().__init__(**kwargs)
        if setup_file:
            pass
        else:
            self.debug = debug
            if self.debug:
                self.num_cpus = 1
                self.num_gpus = 0
                self.local_mode = True
            else:
                self.num_cpus = os.cpu_count() if num_cpus == -1 else num_cpus
                self.num_gpus = num_gpus
                self.local_mode = local_mode
            
            self.restore = restore
            self.framework = framework
        self.training_on = None
        if isinstance(training_on, str):
            if training_on in ENVIRONMENTS:
                self.training_on = ENVIRONMENTS[training_on]()
            elif training_on in DATASETS:
                self.training_on = DATASETS[training_on]
        else:
            self.training_on = training_on
        self.algorithm = ALGORITHMS[algorithm] if isinstance(algorithm, str) else algorithm
        register_trainable(self.algorithm.__name__, self.algorithm)
        self.algorithm_config = ALGORITHMCONFIGS[algorithm_config]() if isinstance(algorithm_config, str) else algorithm_config

        self.policy = POLICIES[policy] if isinstance(policy, str) else policy
        self.policy_config = POLICYCONFIGS[policy_config]() if isinstance(policy_config, str) else policy_config

        self.model = MODELS[model] if isinstance(model, str) else model
        self.model_config = MODELCONFIGS[model_config]() if isinstance(model_config, str) else model_config
        if isinstance(self.algorithm_config, AlgorithmConfig):
            self.algorithm_config.multi_agent(
                    policies = {
                        "default_policy":(
                            self.policy,
                            self.training_on.observation_space(),
                            self.training_on.action_space(),
                            {"config":self.policy_config})},
            )

            self.algorithm_config.training(
                model={
                    'custom_model': self.model.__name__,
                    'custom_model_config':{"config":self.model_config}
                    })
            self.algorithm_config.environment(self.training_on.getName())
        elif isinstance(self.algorithm_config, TrainableConfig):
            self.algorithm_config.update(
                model=self.model,
                model_config=self.model_config,
                dataset=self.training_on,
                num_cpus=self.num_cpus)
            algorithm = tune.with_resources(
                self.algorithm,
                resources={"CPU":self.num_cpus, "GPU":self.num_gpus}
            )
            tune.register_trainable(algorithm.__name__, algorithm)
        self.run_config = run_config
        self.callback = callback

    def getAlgConfig(self) -> AlgorithmConfig:
        return self.algorithm_config

    def run(self):
        """
        Activate model training.

        Based on the preset options of the constructor, a ray server is
        initiated and a new or existing tuning experiment is run. Once complete,
        the ray server is shutdown.

        Args:
        """
        ray.init(
            num_cpus=self.num_cpus,
            num_gpus=math.ceil(self.num_gpus),
            local_mode=self.local_mode,
            # storage="/home/mark/Machine_Learning/Reinforcement_Learning/Chess/results")
            storage="/opt/ray/results")
        
        tuner = None
        if self.restore:
            tuner = tune.Tuner.restore(self.restore, self.algorithm.__name__, param_space=self.algorithm_config, resume_errored=True)
        else:
            tuner = tune.Tuner(
                self.algorithm.__name__,
                run_config=self.run_config,
                param_space=self.algorithm_config,
            )

        tuner.fit()
        ray.shutdown()
        print("Done")

class Serve:
    pass#TODO Incorporate into "Scripts"

    # # from pettingzoo.classic import chess_v5
    # # from starlette.requests import Request
    # # from ray.rllib.algorithms.ppo import PPOConfig
    # # from ray import serve
    # # import requests
    # # import numpy as np
    # # from ray.tune.registry import register_env
    # # from ray.rllib.env import PettingZooEnv


    # def env_creator(args):
    #     env = chess_v5.env()
    #     return env

    # register_env("Chess",lambda config: PettingZooEnv(env_creator(config)))

    # @serve.deployment
    # class ServePPOModel:
    #     def __init__(self, checkpoint_path) -> None:
    #         # Re-create the originally used config.
    #         config = (PPOConfig()
    #                     .framework(framework='torch')
    #                     .rollouts(num_rollout_workers=0)
    #                     .training(model={'dim':8, 'conv_filters':[[20,[1,1],1]]}))
    #         self.algorithm = config.build("Chess")
    #         # Restore the algo's state from the checkpoint.
    #         self.algorithm.restore(checkpoint_path)

    #     async def __call__(self, request: Request):
    #         json_input = await request.json()
    #         # obs = json_input["observation"]

    #         action = self.algorithm.compute_single_action(json_input)
    #         return {"action": int(action)}
        
    # if __name__ == "__main__":
    #     ppo_model = ServePPOModel.bind('./results/test/PPO_Chess_7fb5d_00000_0_2023-05-07_18-27-30/checkpoint_000075')
    #     serve.run(ppo_model)

    #     env = chess_v5.env('human')

    #     for i in range(100):
    #         env.reset()
    #         epRew = 0
    #         done = False
    #         j = 0
    #         while not done:
    #             observation, reward, termination, truncation, info = env.last()
    #             done = termination or truncation
    #             for k,v in observation.items():
    #                 if isinstance(v, np.ndarray):
    #                     observation[k] = v.tolist()
    #             if not done:
    #                 resp = requests.get(
    #                     "http://localhost:8000/", json=observation
    #                 )

    #                 env.step(resp.json()['action'])


if __name__ == "__main__":
    x = ScriptChooser()
    x.complete_run()
    pass