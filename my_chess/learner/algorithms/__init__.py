"""Algorithm to control training of learnable portions of policy"""

from .trainable import Trainable, TrainableConfig, SimpleCustomBatch, collate_wrapper
from .autoencoder import AutoEncoder, AutoEncoderConfig
from .chessevaluation import ChessEvaluation, ChessEvaluationConfig
from .distill import ModelDistill, ModelDistillConfig

from .algorithm import (
    Algorithm,
    AlgorithmConfig,
    measure_accuracy,
    measure_precision,
    measure_recall
    )
from .ppo_cust import PPO, PPOConfig
