from my_chess.learner.models import Model, ModelConfig, ModelAutoEncodable

from ray.rllib.utils.framework import TensorType

import torch
from torch import nn

import math
from copy import copy
from collections import OrderedDict
from typing import (
    Union,
    List)
from typing_extensions import override

class TransformerChessFEConfig(ModelConfig):
    ACTIVATIONS = {
        'relu':nn.ReLU,
        'sigmoid':nn.Sigmoid
    }
    def __init__(
            self,
            hidden_dims:int = 256,
            activation:str = 'relu',
            num_encoder_layers:int = 2,
            num_decoder_layers:int = 2,
            nhead:int = 2,
            batch_norm:bool = True,
            pad_features:bool = False,
            ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.nhead = nhead
        self.batch_norm = batch_norm
        self.pad_features = pad_features
        
    def __str__(self) -> str:
        return "Transformer{}x{}x{}x{}".format(self.hidden_dims, self.nhead, self.num_encoder_layers, self.num_decoder_layers)

class TransformerChessFE(ModelAutoEncodable):
    def __init__(
        self,
        input_sample,
        config:TransformerChessFEConfig = None) -> None:
        super().__init__()
        self.config = config
        self.input_shape = input_sample.shape[-3:]
        self.expected_input_dim = (
            self.input_shape[-1] + self.config.nhead - self.input_shape[-1] % self.config.nhead
            if self.config.pad_features else
            self.config.hidden_dims
        )

        self.body = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.expected_input_dim,
                nhead=self.config.nhead,
                dim_feedforward=self.config.hidden_dims,
                activation=self.config.activation,
                batch_first=True),
            num_layers=self.config.num_encoder_layers,
        )
        class PreProcess(nn.Module):
            def __init__(slf) -> None:
                super().__init__()
                slf.flatten = nn.Flatten(-3,-2)
                slf.inpt_resize = (
                    nn.ZeroPad2d((0,self.expected_input_dim - self.input_shape[-1],0,0))
                    if self.config.pad_features else
                    nn.Linear(self.input_shape[-1], self.expected_input_dim)
                )
                slf.positional_embedding = nn.Embedding(self.input_shape[-3:-1].numel(), self.expected_input_dim)

            def forward(slf, x):
                flt = slf.flatten(x)
                if self.config.pad_features:
                    flt = flt.int()
                flt = slf.inpt_resize(flt)
                pos = slf.positional_embedding(torch.tile(torch.arange(self.input_shape[-3:-1].numel()).to(device=x.device),(flt.shape[0],1)))
                return flt + pos
            
        self.preprocess = PreProcess()
    
    @override
    def __len__(self):
        return len(self.body.layers)
    
    @override
    def __iter__(self):
        return iter(self.body.layers)
    
    @override
    def __getitem__(self, key):
        return nn.Sequential(self.preprocess, *self.body.layers[key])
    
    def forward(
        self,
        input: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        if len(input.shape) < 4:
            input = input.unsqueeze(0)
        prep = self.preprocess(input)
        logits = self.body(prep)
        return logits
    
    def decoder(self):
        class Decoder(nn.Module):
            def __init__(slf) -> None:
                super().__init__()
                # slf.emb = nn.Embedding(self.input_shape.numel(), math.ceil(self.input_shape.numel()**0.5))
                slf.emb = nn.Embedding(self.input_shape[-3:-1].numel(), self.expected_input_dim)
                slf.dec = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=self.expected_input_dim,
                        nhead=self.config.nhead,
                        dim_feedforward=self.config.hidden_dims,
                        activation=self.config.activation,
                        batch_first=True),
                        num_layers=self.config.num_encoder_layers)
                slf.post = nn.Sequential(
                    nn.Linear(self.expected_input_dim, self.input_shape[-1]*2),
                    nn.Unflatten(-2, self.input_shape[-3:-1]),
                    nn.Unflatten(-1, (self.input_shape[-1],2))
                )
            def __getitem__(slf, key):
                cpy = Decoder()
                cpy.emb = slf.emb
                cpy.dec = nn.TransformerDecoder(None,1)
                cpy.dec.layers = slf.dec.layers[key]
                return cpy
            def forward(slf, x):
                slf.to(device=x.device)
                return slf.post(
                    slf.dec(
                        slf.emb(torch.tile(torch.arange(self.input_shape[-3:-1].numel()).to(device=x.device),(x.shape[0],1))),
                        x))
            
        return Decoder()