import argparse
import string
import inspect
import abc
import os
import sys
import math
import time
from typing import (
    Dict,
    Union,
    Literal
)

import ray
from ray import tune, air
from ray.tune.registry import register_trainable

import my_chess.learner.environments
from my_chess.learner.environments import (
    Environment,
    Chess,
)
ENVIRONMENTS = {k:v for k,v in inspect.getmembers(my_chess.learner.environments, inspect.isclass)}

import my_chess.learner.algorithms
from my_chess.learner.algorithms import (
    Algorithm,
    AlgorithmConfig,
    PPO,
    PPOConfig,
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
    ModelConfig
)
MODELS = {k:v for k,v in inspect.getmembers(my_chess.learner.models, inspect.isclass) if not "Config" in k}
MODELCONFIGS = {k:v for k,v in inspect.getmembers(my_chess.learner.models, inspect.isclass) if "Config" in k}

from my_chess.learner.callbacks import DefaultCallbacks


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
            checkpoint=None,
            environment:Union[Environment, str]=None,
            **kwargs) -> None:
        """
        Args:
            checkpoint: directory location of policy to visualize
            environment: Name of environment (if any) to test in.
        """
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.environment = ENVIRONMENTS[environment]() if isinstance(environment, str) else environment
    
    def run(self):
        """
        Args:
        """
        pol1 = pol2 = Policy.from_checkpoint(self.checkpoint)

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

class Train(Script):
    def __init__(
            self,
            setup_file:str=None,
            debug:bool=False,
            restore:str='',
            num_cpus:int=1,
            num_gpus:float=0.,
            local_mode:bool=False,
            environment:Union[Environment, str]=None,
            algorithm:Union[Algorithm, str]=None,
            algorithm_config:Union[AlgorithmConfig, str]=None,
            policy:Union[Policy, str]=None,
            policy_config:Union[PolicyConfig, str]=None,
            model:Union[Model, str]=None,
            model_config:Union[ModelConfig, str]=None,
            run_config:Union[air.RunConfig, str]=None,
            callback:DefaultCallbacks=None,
            framework:Literal["tf","torch"] = "torch",
            **kwargs) -> None:
        """
        Args:
            setup_file: Path to training parameters.
            debug: Boolean determining wether to enable debug mode.
            restore: Tuner checkpoint path from which to continue.
            num_cpus: Number of cpus to allocate to session.
            num_gpus: Number of gpus to allocate to session (including fractional amounts).
            local_mode: Boolean limiting session to be run locally.
            environment: Name of environment (if any) to train in.
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
        self.environment = ENVIRONMENTS[environment]() if isinstance(environment, str) else environment
        self.algorithm = ALGORITHMS[algorithm] if isinstance(algorithm, str) else algorithm
        register_trainable(self.algorithm.__name__, self.algorithm)
        self.algorithm_config = ALGORITHMCONFIGS[algorithm_config]() if isinstance(algorithm_config, str) else algorithm_config

        self.policy = POLICIES[policy] if isinstance(policy, str) else policy
        self.policy_config = POLICYCONFIGS[policy_config]() if isinstance(policy_config, str) else policy_config
        self.algorithm_config.multi_agent(
                policies = {
                    "default_policy":(
                        self.policy,
                        self.environment.observation_space(),
                        self.environment.action_space(),
                        {"config":self.policy_config})},
        )


        self.model = MODELS[model] if isinstance(model, str) else model
        self.model_config = MODELCONFIGS[model_config]() if isinstance(model_config, str) else model_config
        self.algorithm_config.training(
            model={
                'custom_model': self.model.__name__,
                'custom_model_config':{"config":self.model_config}
                })
        self.run_config = run_config
        if self.environment:
            self.algorithm_config.environment(self.environment.getName())
        self.callback = callback

    def getAlgConfig(self) -> AlgorithmConfig:
        return self.algorithm_config

    def run(self):
        """
        Args:
        """
        ray.init(
            num_cpus=self.num_cpus,
            num_gpus=math.ceil(self.num_gpus),
            local_mode=self.local_mode,
            storage="/home/mark/Machine_Learning/Reinforcement_Learning/Chess")
        
        tuner = None
        if self.restore:
            tuner = tune.Tuner.restore(self.restore, self.algorithm.__name__, resume_errored=True)
        else:
            tuner = tune.Tuner(
                self.algorithm.__name__,
                run_config=self.run_config,
                param_space=self.algorithm_config,
            )

        tuner.fit()
        ray.shutdown()
        print("Done")

if __name__ == "__main__":
    x = ScriptChooser()
    x.complete_run()
    pass