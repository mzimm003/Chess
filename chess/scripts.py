import argparse
import string
import inspect
import abc
import os
import sys
from typing import (
    Dict
)

import ray
from ray import tune, air


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
    def __init__(self, **kwargs) -> None:
        """
        Args:
        """
        super().__init__(**kwargs)
    
    def run(self):
        """
        Args:
        """
        pass

class Train(Script):
    def __init__(
            self,
            setup_file:str=None,
            debug:bool=False,
            restore:str='',
            num_cpus:int=1,
            num_gpus:float=0.,
            local_mode:bool=False,
            **kwargs) -> None:
        """
        Args:
            setup_file: Path to training parameters.
            debug: Boolean determining wether to enable debug mode.
            restore: Tuner checkpoint path from which to continue.
            num_cpus: Number of cpus to allocate to session.
            num_gpus: Number of gpus to allocate to session (including fractional amounts).
            local_mode: Boolean limiting session to be run locally.
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

    def run(self):
        """
        Args:
        """
        ray.init(num_cpus=self.num_cpus, num_gpus=self.num_gpus, local_mode=self.local_mode)
        
        tuner = None
        if self.restore:
            tuner = tune.Tuner.restore(self.restore, "PPO", resume_errored=True)
        else:
            tuner = tune.Tuner(
                "PPO",
                run_config=air.RunConfig(
                                local_dir="./results",
                                name="test",
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=25)),
                param_space=PPOConfig()
                                .environment("Chess", env_config={"render_mode":"human"})
                                .multi_agent(
                                    # policy_mapping_fn=policy_mapping_fn,
                                    policy_map_capacity=2,
                                )
                                .callbacks(SelfPlayCallback)
                                .training(
                                    lr=tune.grid_search([0.0001]),
                                    model={
                                        'custom_model': 'my_torch_model',
                                        'custom_model_config': {}
                                        # 'fcnet_hiddens':tune.grid_search([[1280,]]),#,[1280,1280]]),#,[2560,],[2560,2560]]),
                                        # 'dim':8,
                                        # 'conv_filters':[[20,[1,1],1]],
                                        # 'use_attention':True,
                                        # # The number of transformer units within GTrXL.
                                        # # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
                                        # # b) a position-wise MLP.
                                        # "attention_num_transformer_units": 1,
                                        # # The input and output size of each transformer unit.
                                        # "attention_dim": 64,
                                        # # The number of attention heads within the MultiHeadAttention units.
                                        # "attention_num_heads": 1,
                                        # # The dim of a single head (within the MultiHeadAttention units).
                                        # "attention_head_dim": 32,
                                        # # The memory sizes for inference and training.
                                        # "attention_memory_inference": 50,
                                        # "attention_memory_training": 50,
                                        # # The output dim of the position-wise MLP.
                                        # "attention_position_wise_mlp_dim": 32,
                                        # # The initial bias values for the 2 GRU gates within a transformer unit.
                                        # "attention_init_gru_gate_bias": 2.0,
                                        # # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
                                        # "attention_use_n_prev_actions": 0,
                                        # # Whether to feed r_{t-n:t-1} to GTrXL.
                                        # "attention_use_n_prev_rewards": 0,
                                        },
                                    optimizer={'simple_optimizer':True},
                                )
                                # .resources(num_gpus=0.25)
                                .resources(num_gpus=0.85)
                                .framework(framework='torch')
                                .rollouts(num_rollout_workers=2, num_envs_per_worker=50),
            )

        tuner.fit()
        ray.shutdown()
        print("Done")

if __name__ == "__main__":
    x = ScriptChooser()
    x.complete_run()
    pass