from typing import Dict
from ray.train._internal.storage import StorageContext
from ray.tune.logger import Logger
from ray.tune.trainable import Trainable as Trainabletemp
from ray.train.torch import TorchTrainer
from ray.tune.tune import _Config

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

from my_chess.learner.datasets import Dataset
from my_chess.learner.models import Model, ModelConfig

import os

from typing import Any, Callable, Tuple, Type, Union

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

class TrainableConfig(_Config):
    def __init__(
            self,
            num_cpus=None,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:dict=None,
            model:Type[Model]=None,
            model_config:ModelConfig=None,
            batch_size:int=None,
            shuffle:bool=False,
            seed:int=None,
            data_split:Tuple[float, float, float]=None,
            pin_memory:bool=None,
            learning_rate:float=None,
            learning_rate_scheduler:torch.optim.lr_scheduler._LRScheduler=None,
            learning_rate_scheduler_config:dict=None,
            **kwargs) -> None:
        super().__init__()
        self.num_cpus = num_cpus if num_cpus else os.cpu_count()
        self.dataset  = dataset
        self.dataset_config  = dataset_config
        self.optimizer  = optimizer
        self.optimizer_config  = optimizer_config
        self.criterion  = criterion
        self.criterion_config  = criterion_config
        self.model  = model
        self.model_config  = model_config
        self.batch_size  = batch_size
        self.shuffle  = shuffle
        self.seed  = seed
        self.data_split  = data_split
        self.pin_memory  = pin_memory
        self.learning_rate  = learning_rate
        self.learning_rate_scheduler  = learning_rate_scheduler
        self.learning_rate_scheduler_config  = learning_rate_scheduler_config

    def to_dict(self):
        return vars(self)
    
    def getName(self):
        return self.__class__.__name__
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v:
                setattr(self, k, v)

class Trainable(Trainabletemp):
    def __init__(
            self,
            config: Dict[str, Any] = None,
            logger_creator: Callable[[Dict[str, Any]], Logger] = None,
            storage: Union[StorageContext, None] = None
            ):
        super().__init__(config, logger_creator, storage)

    def getName(self):
        return self.__class__.__name__
    
    def setup(self, config:TrainableConfig):
        if isinstance(config, dict):
            config = TrainableConfig(**config)
        self.num_cpus = config.num_cpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = config.dataset(**config.dataset_config)
        self.gen = torch.Generator().manual_seed(config.seed)
        self.trainset, self.valset, self.testset = random_split(self.dataset, config.data_split, generator=self.gen)
        if not config.shuffle:
            # Improves data gathering speeds. Selected indices for each set are still random.
            self.trainset.indices = sorted(self.trainset.indices)
            self.valset.indices = sorted(self.valset.indices)
            self.testset.indices = sorted(self.testset.indices)
        dl_kwargs = dict(
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            collate_fn=collate_wrapper,
            pin_memory=config.pin_memory,
            num_workers=max(config.num_cpus,1),
            prefetch_factor=5
            )
        self.trainloader = DataLoader(self.trainset, **dl_kwargs)
        self.valloader = DataLoader(self.valset, **dl_kwargs)
        self.testloader = DataLoader(self.testset, **dl_kwargs)
        print("Dataloaders created.")
        inp_sample = next(iter(self.trainloader)).inp
        print("Input sample created.")

        self.model = config.model(input_sample=inp_sample, config=config.model_config)
        print("Model created.")

        self.model = self.model.to(self.device)
        print("Models to GPU.")

        self.optimizer = config.optimizer(self.model.parameters(), **config.optimizer_config)
        self.criterion = config.criterion(**config.criterion_config)

        self.learning_rate_scheduler = None
        if not config.learning_rate_scheduler is None:
            self.learning_rate_scheduler = config.learning_rate_scheduler(self.optimizer, **config.learning_rate_scheduler_config)
        return {
            "inp_sample":inp_sample,
            "dl_kwargs":dl_kwargs,
        }
    
    def process_data(self, data):
        inp = data.inp.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)
        target = data.tgt.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)
        return inp, target
    
    def process_output(self, output):
        return output

    def infer_itr(
            self,
            inp,
            target,
            return_acc=False,
            return_prec=False,
            return_recall=False):
        ret = {}
        if self.model.training:
            self.optimizer.zero_grad()
        
        output = self.model(inp)
        
        loss = self.criterion(output, target)

        if self.model.training:
            loss.backward()
            self.optimizer.step()
        ret['loss'] = loss.item()

        result = self.process_output(output)
        if return_acc:
            ret['acc'] = ((result.int() == target.int()).sum()/target.numel()).item()
        if return_prec:
            ret['prec'] = (target.int()[result.int() == 1].sum()/(result.int() == 1).sum()).item()
        if return_recall:
            ret['recall'] = (target.int()[result.int() == 1].sum()/(target.int() == 1).sum()).item()
        return ret


    def step(self):
        self.model.train()
        total_train_loss = 0
        for data in self.trainloader:
            inp, target = self.process_data(data)

            result = self.infer_itr(inp, target)

            total_train_loss += result['loss']

        self.model.eval()
        total_val_loss = 0
        total_acc_ratios = 0
        total_prec_ratios = 0
        total_recall_ratios = 0
        for data in self.valloader:
            inp, target = self.process_data(data)

            result = self.infer_itr(
                inp,
                target,
                return_acc=True,
                return_prec=True,
                return_recall=True)

            total_val_loss += result['loss']
            total_acc_ratios += result['acc']
            total_prec_ratios += result['prec']
            total_recall_ratios += result['recall']
        
        if not self.learning_rate_scheduler is None:
            self.learning_rate_scheduler.step()

        return {
            'model_total_train_loss':total_train_loss,
            'model_mean_train_loss':total_train_loss/(len(self.trainloader)*2),
            'model_total_val_loss':total_val_loss,
            'model_mean_val_loss':total_val_loss/(len(self.valloader)*2),
            'model_mean_val_acc':total_acc_ratios/(len(self.valloader)*2),
            'model_mean_val_prec':total_prec_ratios/(len(self.valloader)*2),
            'model_mean_val_recall':total_recall_ratios/(len(self.valloader)*2),
        }