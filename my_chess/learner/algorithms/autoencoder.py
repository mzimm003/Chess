from typing import Optional, Type, Tuple, Callable
from types import SimpleNamespace
import inspect
from collections.abc import Iterable
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
from my_chess.learner.datasets import Dataset, ChessData
from my_chess.learner.models import Model, ModelConfig, ModelAutoEncodable
import my_chess.learner.algorithms as algorithms

class AutoEncoderConfig(TrainableConfig):
    def __init__(
            self,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:Callable=None,
            model:Type[ModelAutoEncodable]=None,
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
        self.dataset = dataset if dataset else ChessData
        self.dataset_config = dataset_config if dataset_config else {"dataset_dir":"/opt/datasets/Chess-CCRL-404"}
        self.optimizer = optimizer if optimizer else Adam
        self.optimizer_config = optimizer_config if optimizer_config else {"lr":learning_rate}
        self.criterion = criterion if criterion else nn.MSELoss
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
            model:Type[ModelAutoEncodable]=None,
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
        
class AutoEncoder(Trainable):
    # @staticmethod
    # def layer_step(config):
    #     config = SimpleNamespace(config)
    #     idx = config.idx
    #     layer = config.layer
    #     layer = prepare_model(layer)

    #     optimizer = config.optimizer_class(layer.parameters(), **config.optimizer_config)
    #     optimizer = prepare_optimizer(optimizer)

    #     criterion = config.criterion

    #     trainloader = prepare_data_loader(config.trainloader)
    #     valloader = prepare_data_loader(config.valloader)

    #     layer.train()

    #     total_train_loss = 0
    #     for data in trainloader:
    #         inp = data[0]
    #         target = data[0]
    #         # inp = inp.to(dtype=next(iter(layer.parameters())).dtype , device=config.device)
    #         # target = target.to(dtype=next(iter(layer.parameters())).dtype, device=config.device)

    #         output = layer(inp)
            
    #         optimizer.zero_grad()
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()

    #         total_train_loss += loss.item()
        
    #     layer.eval()
    #     total_val_loss = 0
    #     for data in valloader:
    #         inp = data[0]
    #         target = data[0]
    #         output = layer(inp)

    #         loss = criterion(output, target)

    #         total_val_loss += loss.item()

    #     ray.train.report(
    #         metrics = {
    #         'lyr_{}_total_train_loss'.format(idx):total_train_loss,
    #         'lyr_{}_mean_train_loss'.format(idx):total_train_loss/len(trainloader),
    #         'lyr_{}_total_val_loss'.format(idx):total_val_loss,
    #         'lyr_{}_mean_val_loss'.format(idx):total_val_loss/len(valloader),
    #         }
    #     )

    def setup(self, config:AutoEncoderConfig):
        if isinstance(config, dict):
            config = AutoEncoderConfig(**config)
        self.dataset = config.dataset(**config.dataset_config)
        gen = torch.Generator().manual_seed(config.seed)
        self.trainset, self.valset, self.testset = random_split(self.dataset, config.data_split, generator=gen)
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
        self.model_decoder = self.create_decoder(self.model)
        print("Decoder created.")

        self.num_cpus = config.num_cpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model_decoder = self.model_decoder.to(self.device)
        print("Models to GPU.")

        # self.optimizer = None
        # self.optimizer_class = config.optimizer
        # self.optimizer_config = config.optimizer_config
        self.optimizer = config.optimizer(self.model.parameters(), **config.optimizer_config)
        self.criterion = config.criterion(**config.criterion_config)

        self.learning_rate_scheduler = None
        if not config.learning_rate_scheduler is None:
            self.learning_rate_scheduler = config.learning_rate_scheduler(self.optimizer, **config.learning_rate_scheduler_config)
        # self.learning_rate_scheduler_config = config.learning_rate_scheduler_config

        self.idx = 0
        self.layer = None

        # self.train_config = {
        #                 "idx" : self.idx,
        #                 "layer" : self.layer,
        #                 "optimizer_class" : self.optimizer_class,
        #                 "optimizer_config" : self.optimizer_config,
        #                 "trainloader" : self.trainloader,
        #                 "valloader" : self.valloader,
        #                 "criterion" : self.criterion,
        #             },
    

    @staticmethod
    def create_decoder(mod):
        return mod.decoder()

    # def step(self):
    #     losses = {}
    #     for i, lyr in enumerate(self.model.ff):
    #         if i <= len(self.model.ff)//2:
    #             self.idx = i
    #             i = i*2
    #             partial_model = nn.Sequential(self.model.flatten, self.model.ff[:i+2], self.model_decoder[-(i+3):])
    #             self.layer = partial_model
    #             trainer = TorchTrainer(
    #                 train_loop_per_worker=AutoEncoder.layer_step,
    #                 train_loop_config=self.train_config,
    #                 scaling_config=ScalingConfig(num_workers=self.num_cpus-1, use_gpu=self.device == "cuda", resources_per_worker={"CPU":1, "GPU":1/(self.num_cpus-1)-.000000001}),
    #                 )
    #             result = trainer.fit()
    #             losses.update(vars(result))
    #     return losses

    def step(self):
        losses = {}
        full_model = False
        for i, lyr in enumerate(self.model):
            if i == len(self.model)-1:
                full_model = True
            partial_model = nn.Sequential(self.model[:i+1], self.model_decoder[-(i+1):])

            optimizer = self.optimizer.__class__(partial_model.parameters())
            state_dict = self.optimizer.state_dict()
            state_dict['state'] = {}
            state_dict['param_groups'][0]['params'] = optimizer.state_dict()['param_groups'][0]['params']
            optimizer.load_state_dict(state_dict=state_dict)
            
            losses.update(self.layer_step(i, partial_model, optimizer, full_model=full_model))
        if not self.learning_rate_scheduler is None:
            self.learning_rate_scheduler.step()
        return losses

    def layer_step(self, idx, layer, optimizer, full_model=False):
        layer.train()
        i = 0
        total_train_loss = 0
        # print("Total ITERATIONS: {}".format(len(self.trainloader)))
        for data in self.trainloader:
            temp = data.inp.to(device=str(self.device), dtype=next(iter(layer.parameters())).dtype)
            inpt = temp
            target = torch.stack([temp, 1-temp], -1)
            optimizer.zero_grad()

            output = layer(inpt)
            loss = self.criterion(output, target)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            i += 1
            # print("Made it to ITERATION: {}".format(i))

        layer.eval()
        total_val_loss = 0
        
        if full_model:
            # TODO put accuracy/recall/whatever here
            total_acc_ratios = 0
            total_prec_ratios = 0
            total_recall_ratios = 0
            just_board_metrics = {i:{"ttl_acc":0,"ttl_prec":0,"ttl_rcl":0} for i in range(7)}

            for data in self.valloader:
                temp = data.inp.to(device=str(self.device), dtype=next(iter(layer.parameters())).dtype)
                inpt = temp
                target = torch.stack([temp, 1-temp], -1)

                output = layer(inpt)
                loss = self.criterion(output, target)
                total_val_loss += loss.item()
                
                board_result = output.max(-1).indices
                target = target.max(-1).indices
                total_acc_ratios += algorithms.measure_accuracy(board_result.int(), target.int(), 0).item()
                total_prec_ratios += algorithms.measure_precision(board_result.int(), target.int(), 0).item()
                total_recall_ratios += algorithms.measure_recall(board_result.int(), target.int(), 0).item()
                for i, board in just_board_metrics.items():
                    b_r = board_result[...,7+13*i:19+13*i,:]
                    inp = target[...,7+13*i:19+13*i,:]
                    board["ttl_acc"] += algorithms.measure_accuracy(b_r.int(), inp.int(), 0).item()
                    board["ttl_prec"] += algorithms.measure_precision(b_r.int(), inp.int(), 0).item()
                    board["ttl_rcl"] += algorithms.measure_recall(b_r.int(), inp.int(), 0).item()
            ret_dict = {
                'model_total_train_loss'.format(idx):total_train_loss,
                'model_mean_train_loss'.format(idx):total_train_loss/len(self.trainloader),
                'model_total_val_loss'.format(idx):total_val_loss,
                'model_mean_val_loss'.format(idx):total_val_loss/len(self.valloader),
                'model_mean_val_acc'.format(idx):total_acc_ratios/len(self.valloader),
                'model_mean_val_prec'.format(idx):total_prec_ratios/len(self.valloader),
                'model_mean_val_recall'.format(idx):total_recall_ratios/len(self.valloader),
            }
            for i, board in just_board_metrics.items():
                ret_dict.update(
                    {
                        'model_board_{}_mean_val_acc'.format(i):board["ttl_acc"]/len(self.valloader),
                        'model_board_{}_mean_val_prec'.format(i):board["ttl_prec"]/len(self.valloader),
                        'model_board_{}_mean_val_recall'.format(i):board["ttl_rcl"]/len(self.valloader),
                    }
                )
            return ret_dict
        else:
            for data in self.valloader:
                temp = data.inp.to(device=str(self.device), dtype=next(iter(layer.parameters())).dtype)
                inpt = temp
                target = torch.stack([temp, 1-temp], -1)

                output = layer(inpt)
                loss = self.criterion(output, target)
                total_val_loss += loss.item()
            return {
                'lyr_{}_total_train_loss'.format(idx):total_train_loss,
                'lyr_{}_mean_train_loss'.format(idx):total_train_loss/len(self.trainloader),
                'lyr_{}_total_val_loss'.format(idx):total_val_loss,
                'lyr_{}_mean_val_loss'.format(idx):total_val_loss/len(self.valloader),
                }

    def save_checkpoint(self, checkpoint_dir):
        # Save model and optimizer state
        torch.save(self.model.state_dict(), checkpoint_dir + "/model.pt")
        torch.save(self.model_decoder.state_dict(), checkpoint_dir + "/model_decoder.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_dir + "/optimizer.pt")

    def load_checkpoint(self, checkpoint_dir):
        # Load model and optimizer state
        self.model.load_state_dict(torch.load(checkpoint_dir + "/model.pt"))
        self.model_decoder.load_state_dict(torch.load(checkpoint_dir + "/model_decoder.pt"))
        self.optimizer.load_state_dict(torch.load(checkpoint_dir + "/optimizer.pt"))

    def cleanup(self):
        # Free resources
        del self.model, self.model_decoder, self.optimizer