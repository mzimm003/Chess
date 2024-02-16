from typing import Optional, Type, Tuple, Callable
from types import SimpleNamespace
import inspect

from ray.rllib.algorithms.ppo import PPO as PPOtemp
from ray.rllib.utils.annotations import override
from ray.train.torch import TorchTrainer, prepare_model, prepare_optimizer, prepare_data_loader
from ray.train import RunConfig, ScalingConfig
import ray.train
from torch.utils.data import DataLoader, random_split
from torch.optim import Optimizer, Adam
import torch
from torch import nn

from my_chess.learner.algorithms import Trainable, TrainableConfig
from my_chess.learner.policies import Policy, PPOPolicy
from my_chess.learner.datasets import Dataset, ChessData
from my_chess.learner.models import Model, ModelConfig

class AutoEncoderConfig(TrainableConfig):
    def __init__(
            self,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:Callable=None,
            model:Model=None,
            model_config:ModelConfig=None,
            batch_size:int=64,
            shuffle:bool=False, #creates slow down given data separation between files
            seed:int=42,
            data_split:Tuple[float, float, float]=(0.025, 0.005, 0.97),
            pin_memory:bool=True,
            **kwargs
            ) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset if dataset else ChessData
        self.dataset_config = dataset_config if dataset_config else {"data_dir":"/home/mark/Machine_Learning/Reinforcement_Learning/Chess/Data/Chess-CCRL-404"}
        self.optimizer = optimizer if optimizer else Adam
        self.optimizer_config = optimizer_config if optimizer_config else {}
        self.criterion = criterion if criterion else nn.MSELoss
        self.criterion_config = criterion_config if criterion_config else {}
        self.model = model
        self.model_config = model_config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.data_split = data_split
        self.pin_memory = pin_memory
    
    def update(
            self,
            dataset:Dataset=None,
            dataset_config:dict=None,
            optimizer:Optimizer=None,
            optimizer_config:dict=None,
            criterion:Callable=None,
            criterion_config:Callable=None,
            model:Model=None,
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
    @staticmethod
    def layer_step(config):
        config = SimpleNamespace(config)
        idx = config.idx
        layer = config.layer
        layer = prepare_model(layer)

        optimizer = config.optimizer_class(layer.parameters(), **config.optimizer_config)
        optimizer = prepare_optimizer(optimizer)

        criterion = config.criterion

        trainloader = prepare_data_loader(config.trainloader)
        valloader = prepare_data_loader(config.valloader)

        layer.train()

        total_train_loss = 0
        for data in trainloader:
            inp = data[0]
            target = data[0]
            # inp = inp.to(dtype=next(iter(layer.parameters())).dtype , device=config.device)
            # target = target.to(dtype=next(iter(layer.parameters())).dtype, device=config.device)

            output = layer(inp)
            
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        layer.eval()
        total_val_loss = 0
        for data in valloader:
            inp = data[0]
            target = data[0]
            output = layer(inp)

            loss = criterion(output, target)

            total_val_loss += loss.item()

        ray.train.report(
            metrics = {
            'lyr_{}_total_train_loss'.format(idx):total_train_loss,
            'lyr_{}_mean_train_loss'.format(idx):total_train_loss/len(trainloader),
            'lyr_{}_total_val_loss'.format(idx):total_val_loss,
            'lyr_{}_mean_val_loss'.format(idx):total_val_loss/len(valloader),
            }
        )

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
            pin_memory=config.pin_memory,
            num_workers=config.num_cpus
            )
        self.trainloader = DataLoader(self.trainset, **dl_kwargs)
        self.valloader = DataLoader(self.valset, **dl_kwargs)
        self.testloader = DataLoader(self.testset, **dl_kwargs)
        inp_sample = next(iter(self.trainloader))[0]
        
        self.model = config.model(input_sample=inp_sample, config=config.model_config)
        self.model_decoder = self.create_decoder(self.model, inp_sample)

        self.num_cpus = config.num_cpus
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.optimizer = None
        self.optimizer_class = config.optimizer
        self.optimizer_config = config.optimizer_config
        self.criterion = config.criterion(**config.criterion_config)

        self.idx = 0
        self.layer = None

        self.train_config = {
                        "idx" : self.idx,
                        "layer" : self.layer,
                        "optimizer_class" : self.optimizer_class,
                        "optimizer_config" : self.optimizer_config,
                        "trainloader" : self.trainloader,
                        "valloader" : self.valloader,
                        "criterion" : self.criterion,
                    },
    


    def create_decoder(self, mod, input_sample):
        dec = []
        post_attach = None
        for lyr in reversed(mod.ff):
            init_dict = {parm:val for parm,val in lyr.__dict__.items() if parm in inspect.signature(lyr.__init__).parameters}
            if "in_features" in init_dict:
                temp = init_dict["in_features"]
                init_dict["in_features"] = init_dict["out_features"]
                init_dict["out_features"] = temp
            dec.append(lyr.__class__(**init_dict))
        dec.append(nn.Unflatten(-1, input_sample.shape[1:]))
        return nn.Sequential(*dec)

    def step(self):
        losses = {}
        for i, lyr in enumerate(self.model.ff):
            if i <= len(self.model.ff)//2:
                self.idx = i
                i = i*2
                partial_model = nn.Sequential(self.model.flatten, self.model.ff[:i+2], self.model_decoder[-(i+3):])
                self.layer = partial_model
                trainer = TorchTrainer(
                    train_loop_per_worker=AutoEncoder.layer_step,
                    train_loop_config=self.train_config,
                    scaling_config=ScalingConfig(num_workers=self.num_cpus-1, use_gpu=self.device == "cuda", resources_per_worker={"CPU":1, "GPU":1/(self.num_cpus-1)-.000000001}),
                    )
                result = trainer.fit()
                losses.update(vars(result))
        return losses


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