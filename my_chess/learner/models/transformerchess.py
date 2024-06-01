from my_chess.learner.models import Model, ModelConfig

from ray.rllib.utils.framework import TensorType

import torch
from torch import nn

import math
from typing import (
    Union,
    List)

class TransformerChessFEConfig(ModelConfig):
    ACTIVATIONS = {
        'relu':nn.ReLU,
        'sigmoid':nn.Sigmoid
    }
    def __init__(
            self,
            hidden_dims:int = 2048,
            activation:str = 'relu',
            num_encoder_layers:int = 2,
            num_decoder_layers:int = 2,
            nhead:int = 3,
            batch_norm:bool = True
            ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.nhead = nhead
        self.batch_norm = batch_norm
        
    def __str__(self) -> str:
        return "Transformer{}x{}x{}x{}".format(self.hidden_dims, self.nhead, self.num_encoder_layers, self.num_decoder_layers)

class TransformerChessFE(Model):
    def __init__(
        self,
        input_sample,
        config:TransformerChessFEConfig = None) -> None:
        super().__init__()
        self.config = config
        self.input_shape = input_sample.shape[-3:]
        self.positional_embedding = nn.Embedding(input_sample.shape[-3:-1].numel(), input_sample.shape[-1])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_sample.shape[-1],
                nhead=self.config.nhead,
                dim_feedforward=self.config.hidden_dims,
                activation=self.config.activation,
                batch_first=True),
            num_layers=self.config.num_encoder_layers,
        )
        self.flatten = nn.Flatten(-3,-2)
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        if len(input.shape) < 4:
            input = input.unsqueeze(0)
        flt = self.flatten(input)
        pos = self.positional_embedding(torch.arange(flt.shape[-2]))
        logits = self.encoder(flt+pos)
        return logits
    
    def decoder(self):
        class Decoder(nn.Module):
            def __init__(slf) -> None:
                super().__init__()
                slf.emb = nn.Embedding(self.input_shape.numel(), math.ceil(self.input_shape.numel()**0.5))
                slf.dec = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=self.input_sample.shape[-1],
                        nhead=self.config.nhead,
                        dim_feedforward=self.config.hidden_dims,
                        activation=self.config.activation,
                        batch_first=True),
                        num_layers=self.config.num_encoder_layers)
            def forward(slf, x):
                return slf.dec(slf.emb(torch.arange(self.input_shape.numel())), x)
            
        dec = []
        postprocess = []
        hid_dims = list(reversed(self.config.hidden_dims))
        actvs = list(reversed(self.config.activations))
        for i, lyr_dim in enumerate(hid_dims):
            lyr = []
            post_processing = []
            if i == len(hid_dims)-1:
                lyr.append(nn.Linear(lyr_dim, self.input_shape.numel()*2))
            else:
                lyr.append(nn.Linear(lyr_dim, hid_dims[i+1]))
            if i < len(hid_dims)-1:
                if self.config.batch_norm:
                    post_processing.append(nn.BatchNorm1d(hid_dims[i+1]))
                post_processing.append(actvs[i]())
            lyr.append(nn.Sequential(*post_processing))
            dec.append(nn.Sequential(*lyr))
        postprocess.append(nn.Unflatten(-1, (*self.input_shape, 2)))
        postprocess.append(nn.Softmax(-1))
        return nn.Sequential(
            OrderedDict([('body',nn.Sequential(*dec)),
                         ('postprocess',nn.Sequential(*postprocess))]))