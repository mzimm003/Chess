"""Models to learn optimal actions based on environment states"""

from .model import ModelRLLIB, ModelRRLIBConfig, Model, ModelConfig
from .tobenamed import ToBeNamed, ToBeNamedConfig
from .Qlearner import QLearner, QLearnerConfig
from .deepchess import (
        DeepChessFE,
        DeepChessFEConfig,
        DeepChessEvaluator,
        DeepChessEvaluatorConfig,
        DeepChessAlphaBeta,
        DeepChessAlphaBetaConfig,
        DeepChessRL,
        DeepChessRLConfig,)
from .transformerchess import(
        TransformerChessFE,
        TransformerChessFEConfig
)

import inspect
import sys
from ray.rllib.models import ModelCatalog
reg_models = {k:v for k,v in inspect.getmembers(sys.modules[__name__], inspect.isclass) if not "Config" in k}
for mod_name, mod in reg_models.items():
        ModelCatalog.register_custom_model(mod_name, mod)