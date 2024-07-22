from typing import Optional, Type, Tuple, Callable
from types import SimpleNamespace
import inspect
import shutil

from ray.rllib.algorithms.ppo import PPO as PPOtemp
from ray.rllib.utils.annotations import override
from ray.train.torch import TorchTrainer, prepare_model, prepare_optimizer, prepare_data_loader
from ray.train import RunConfig, ScalingConfig
import ray.train
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer, Adam
import torch
from torch import nn

from my_chess.learner.algorithms import Trainable, TrainableConfig, collate_wrapper
from my_chess.learner.policies import Policy, PPOPolicy
from my_chess.learner.datasets import Dataset, ChessDataWinLossPairs
from my_chess.learner.models import Model, ModelConfig
from my_chess.learner.environments import Chess

class ChessEvaluationConfig(TrainableConfig):
    def __init__(
            self,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:Callable=None,
            model:Type[Model]=None,
            model_config:ModelConfig=None,
            batch_size:int=128,
            shuffle:bool=False, #True creates slow down given data separation between files, and can also cause RAM to blow up
            seed:int=42,
            data_split:Tuple[float, float, float]=(0.225, 0.025, 0.75),
            pin_memory:bool=True,
            learning_rate:float=0.0001,
            learning_rate_scheduler:torch.optim.lr_scheduler._LRScheduler=None,
            learning_rate_scheduler_config:dict=None,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset if dataset else ChessDataWinLossPairs
        self.dataset_config = dataset_config if dataset_config else {"dataset_dir":"/opt/datasets/Chess-CCRL-404"}
        self.optimizer = optimizer if optimizer else Adam
        self.optimizer_config = optimizer_config if optimizer_config else {"lr":learning_rate}
        self.criterion = criterion if criterion else nn.CrossEntropyLoss
        self.criterion_config = criterion_config if criterion_config else {}
        self.model = model
        self.model_config = model_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.data_split = data_split
        self.pin_memory = pin_memory
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_config = learning_rate_scheduler_config
    
    def update(
            self,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:Callable=None,
            model:Type[Model]=None,
            model_config:ModelConfig=None,
            batch_size:int=None,
            shuffle:bool=None,
            seed:int=None,
            data_split:Tuple[float, float, float]=None,
            pin_memory:bool=None,
            **kwargs
            ) -> None:
        super().update(
            dataset = dataset,
            dataset_config = dataset_config,
            optimizer = optimizer,
            optimizer_config = optimizer_config,
            criterion = criterion,
            criterion_config = criterion_config,
            model = model,
            model_config = model_config,
            batch_size = batch_size,
            shuffle = shuffle,
            seed = seed,
            data_split = data_split,
            pin_memory = pin_memory,
            **kwargs
        )
        
class ChessEvaluation(Trainable):
    def setup(self, config:ChessEvaluationConfig):
        if isinstance(config, dict):
            config = ChessEvaluationConfig(**config)
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
            prefetch_factor=2
            )
        self.trainloader = DataLoader(self.trainset, **dl_kwargs)
        self.valloader = DataLoader(self.valset, **dl_kwargs)
        self.testloader = DataLoader(self.testset, **dl_kwargs)
        print("Dataloaders created.")
        inp_sample = next(iter(self.trainloader)).inp
        print("Input sample created.")
        
        self.model = config.model(input_sample=inp_sample, config=config.model_config)
        print("Model created.")

        self.num_cpus = config.num_cpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        print("Models to GPU.")

        self.optimizer = config.optimizer(self.model.parameters(), **config.optimizer_config)
        self.criterion = config.criterion(**config.criterion_config)

        self.learning_rate_scheduler = None
        if not config.learning_rate_scheduler is None:
            self.learning_rate_scheduler = config.learning_rate_scheduler(self.optimizer, **config.learning_rate_scheduler_config)

    def step(self):
        self.model.train()
        total_train_loss = 0
        for data in self.trainloader:
            # Train once on original observation and result
            inp = data.inp.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)
            target = data.tgt.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)
            
            self.optimizer.zero_grad()

            output = self.model(inp)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()
            
            # Train once more on swapped observation and opposite result
            inp = inp.flip(-4)
            target = 1 - data.tgt.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)
            self.optimizer.zero_grad()

            output = self.model(inp)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_train_loss += loss.item()

        self.model.eval()
        total_val_loss = 0
        
        total_acc_ratios = 0
        total_prec_ratios = 0
        total_recall_ratios = 0

        for data in self.valloader:
            # To ensure balanced wins and losses, follow same process as training
            # Validate once on original observation and result
            inp = data.inp.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)
            target = data.tgt.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)

            output = self.model(inp)
            loss = self.criterion(output, target)
            total_val_loss += loss.item()
            
            result = torch.round(output)
            total_acc_ratios += ((result.int() == target.int()).sum()/target.numel()).item()
            total_prec_ratios += (target.int()[result.int() == 1].sum()/(result.int() == 1).sum()).item()
            total_recall_ratios += (target.int()[result.int() == 1].sum()/(target.int() == 1).sum()).item()

            # Validate once more on swapped observation and opposite result
            inp = inp.flip(-4)
            target = 1 - data.tgt.to(device=str(self.device), dtype=next(iter(self.model.parameters())).dtype)

            output = self.model(inp)
            loss = self.criterion(output, target)
            total_val_loss += loss.item()
            
            result = torch.round(output)
            total_acc_ratios += ((result.int() == target.int()).sum()/target.numel()).item()
            total_prec_ratios += (target.int()[result.int() == 1].sum()/(result.int() == 1).sum()).item()
            total_recall_ratios += (target.int()[result.int() == 1].sum()/(target.int() == 1).sum()).item()
        
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


    def save_checkpoint(self, checkpoint_dir):
        # Save model and optimizer state
        torch.save(self.model.state_dict(), checkpoint_dir + "/model.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_dir + "/optimizer.pt")

    def load_checkpoint(self, checkpoint_dir):
        # Load model and optimizer state
        self.model.load_state_dict(torch.load(checkpoint_dir + "/model.pt"))
        self.optimizer.load_state_dict(torch.load(checkpoint_dir + "/optimizer.pt"))

    def cleanup(self):
        # Free resources
        del self.model, self.optimizer
