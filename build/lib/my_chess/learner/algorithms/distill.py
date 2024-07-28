from my_chess.learner.algorithms import Trainable, TrainableConfig, collate_wrapper
from my_chess.learner.datasets import Dataset, ChessDataWinLossPairs
from my_chess.learner.models import Model, ModelConfig

from typing import Callable, Type, Tuple, Union
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, random_split

class ModelDistillConfig(TrainableConfig):
    def __init__(
            self,
            num_cpus=None,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:dict=None,
            distill_criterion:Callable=None,
            distill_criterion_config:dict=None,
            model:Type[Model]=None,
            model_config:ModelConfig=None,
            parent_model:Type[Model]=None,
            parent_model_config:ModelConfig=None,
            parent_model_param_dir:Union[str, Path]=None,
            batch_size:int=128,
            shuffle:bool=False, #True creates slow down given data separation between files, and can also cause RAM to blow up
            seed:int=42,
            data_split:Tuple[float, float, float]=(0.225, 0.025, 0.75),
            pin_memory:bool=True,
            learning_rate:float=0.0001,
            learning_rate_scheduler:torch.optim.lr_scheduler._LRScheduler=None,
            learning_rate_scheduler_config:dict=None,
            train_on_teacher_only:bool=False,
            **kwargs
            ) -> None:
        super().__init__(
            num_cpus = num_cpus,
            dataset = dataset if dataset else ChessDataWinLossPairs,
            dataset_config = dataset_config if dataset_config else {"dataset_dir":"/opt/datasets/Chess-CCRL-404"},
            optimizer = optimizer if optimizer else Adam,
            optimizer_config = optimizer_config if optimizer_config else {"lr":learning_rate},
            criterion = criterion if criterion else nn.CrossEntropyLoss,
            criterion_config = criterion_config if criterion_config else {},
            model = model,
            model_config = model_config,
            batch_size = batch_size,
            shuffle = shuffle,
            seed = seed,
            data_split = data_split,
            pin_memory = pin_memory,
            learning_rate = learning_rate,
            learning_rate_scheduler = learning_rate_scheduler,
            learning_rate_scheduler_config = learning_rate_scheduler_config,
            **kwargs)
        self.distill_criterion = distill_criterion if distill_criterion else nn.MSELoss
        self.distill_criterion_config = distill_criterion_config if distill_criterion_config else {}
        self.parent_model = parent_model
        self.parent_model_config = parent_model_config
        self.parent_model_param_dir = parent_model_param_dir
        self.train_on_teacher_only = train_on_teacher_only
    
    def update(
            self,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:dict=None,
            distill_criterion:Callable=None,
            distill_criterion_config:dict=None,
            model:Type[Model]=None,
            model_config:ModelConfig=None,
            parent_model:Type[Model]=None,
            parent_model_config:ModelConfig=None,
            parent_model_param_dir:Union[str, Path]=None,
            batch_size:int=None,
            shuffle:bool=None, #True creates slow down given data separation between files, and can also cause RAM to blow up
            seed:int=None,
            data_split:Tuple[float, float, float]=None,
            pin_memory:bool=None,
            learning_rate:float=None,
            learning_rate_scheduler:torch.optim.lr_scheduler._LRScheduler=None,
            learning_rate_scheduler_config:dict=None,
            train_on_teacher_only:bool=None,
            **kwargs
            ) -> None:
        super().update(
            dataset = dataset,
            dataset_config = dataset_config,
            optimizer = optimizer,
            optimizer_config = optimizer_config,
            criterion = criterion,
            criterion_config = criterion_config,
            distill_criterion = distill_criterion,
            distill_criterion_config = distill_criterion_config,
            model = model,
            model_config = model_config,
            parent_model = parent_model,
            parent_model_config = parent_model_config,
            parent_model_param_dir = parent_model_param_dir,
            batch_size = batch_size,
            shuffle = shuffle,
            seed = seed,
            data_split = data_split,
            pin_memory = pin_memory,
            learning_rate = learning_rate,
            learning_rate_scheduler = learning_rate_scheduler,
            learning_rate_scheduler_config = learning_rate_scheduler_config,
            train_on_teacher_only=train_on_teacher_only,
            **kwargs
        )
        
class ModelDistill(Trainable):
    def setup(self, config:ModelDistillConfig):
        if isinstance(config, dict):
            config = ModelDistillConfig(**config)
        super_locals = super().setup(config=config)

        self.parent_model = config.parent_model(input_sample=super_locals["inp_sample"], config=config.parent_model_config)
        if config.parent_model_param_dir:
            self.parent_model.load_state_dict(torch.load(config.parent_model_param_dir))
        self.parent_model.to(device=self.device)
        self.parent_model.eval()
        self.distill_criterion = config.distill_criterion(**config.distill_criterion_config)
        print("Parent Model created.")
        self.train_on_teacher_only = config.train_on_teacher_only

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
        parent_output = self.parent_model(inp)
        
        class_loss = 0
        if not self.train_on_teacher_only:
            class_loss += self.criterion(output, target)
        distill_loss = self.distill_criterion(output, parent_output)

        loss = class_loss + distill_loss

        if self.model.training:
            loss.backward()
            self.optimizer.step()
        if not self.train_on_teacher_only:
            ret['loss_class'] = class_loss.item()
        else:
            ret['loss_class'] = class_loss
        ret['loss_distill'] = distill_loss.item()

        result = self.process_output(output)
        if not self.train_on_teacher_only:
            if return_acc:
                ret['acc'] = ((result.int() == target.int()).sum()/target.numel()).item()
            if return_prec:
                ret['prec'] = (target.int()[result.int() == 1].sum()/(result.int() == 1).sum()).item()
            if return_recall:
                ret['recall'] = (target.int()[result.int() == 1].sum()/(target.int() == 1).sum()).item()
        return ret
    
    def process_output(self, output):
        return torch.round(output)

    def step(self):
        self.model.train()
        total_train_loss = 0
        total_train_class_loss = 0
        total_train_distill_loss = 0
        for data in self.trainloader:
            # Train once on original observation and result
            inp, target = self.process_data(data)
            
            result = self.infer_itr(inp, target)

            total_train_loss += result['loss_class'] + result['loss_distill']
            total_train_class_loss += result['loss_class']
            total_train_distill_loss += result['loss_distill']
            
            # Train once more on swapped observation and opposite result
            inp = inp.flip(-4)
            target = 1 - target

            result = self.infer_itr(inp, target)

            total_train_loss += result['loss_class'] + result['loss_distill']
            total_train_class_loss += result['loss_class']
            total_train_distill_loss += result['loss_distill']

        self.model.eval()
        total_val_loss = 0
        total_val_class_loss = 0
        total_val_distill_loss = 0
        
        total_acc_ratios = 0
        total_prec_ratios = 0
        total_recall_ratios = 0

        for data in self.valloader:
            # To ensure balanced wins and losses, follow same process as training
            # Validate once on original observation and result
            inp, target = self.process_data(data)
            
            result = self.infer_itr(
                inp,
                target,
                return_acc=True,
                return_prec=True,
                return_recall=True)

            total_val_loss += result['loss_class'] + result['loss_distill']
            total_val_class_loss += result['loss_class']
            total_val_distill_loss += result['loss_distill']
            if not self.train_on_teacher_only:
                total_acc_ratios += result['acc']
                total_prec_ratios += result['prec']
                total_recall_ratios += result['recall']

            # Validate once more on swapped observation and opposite result
            inp = inp.flip(-4)
            target = 1 - target
            
            result = self.infer_itr(
                inp,
                target,
                return_acc=True,
                return_prec=True,
                return_recall=True)

            total_val_loss += result['loss_class'] + result['loss_distill']
            total_val_class_loss += result['loss_class']
            total_val_distill_loss += result['loss_distill']
            if not self.train_on_teacher_only:
                total_acc_ratios += result['acc']
                total_prec_ratios += result['prec']
                total_recall_ratios += result['recall']
        
        if not self.learning_rate_scheduler is None:
            self.learning_rate_scheduler.step()

        return {
            'model_total_train_loss':total_train_loss,
            'model_mean_train_loss':total_train_loss/(len(self.trainloader)*2),
            'model_total_train_class_loss':total_train_class_loss,
            'model_mean_train_class_loss':total_train_class_loss/(len(self.trainloader)*2),
            'model_total_train_distill_loss':total_train_distill_loss,
            'model_mean_train_distill_loss':total_train_distill_loss/(len(self.trainloader)*2),
            'model_total_val_loss':total_val_loss,
            'model_mean_val_loss':total_val_loss/(len(self.valloader)*2),
            'model_total_val_class_loss':total_val_class_loss,
            'model_mean_val_class_loss':total_val_class_loss/(len(self.valloader)*2),
            'model_total_val_distill_loss':total_val_distill_loss,
            'model_mean_val_distill_loss':total_val_distill_loss/(len(self.valloader)*2),
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
        del self.model, self.optimizer, self.parent_model