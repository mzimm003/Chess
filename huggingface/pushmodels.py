import pickle
from pathlib import Path
from typing import Dict, Type, Union, Literal, List

import torch

from my_chess.learner.models import (
    Model,
    ModelConfig,
    DeepChessAlphaBeta,
    DeepChessAlphaBetaConfig,
    DeepChessEvaluator,
    DeepChessEvaluatorConfig,
    DeepChessFE,
    DeepChessFEConfig
)
from huggingface_hub import PyTorchModelHubMixin

class DCFE(DeepChessFE, PyTorchModelHubMixin):
    def __init__(
            self,
            hidden_dims:Union[int, List[int]]=[4096, 1024, 256, 128],
            activations:Union[str, List[str]]='relu',
            batch_norm:bool = True) -> None:
        input_sample = torch.rand(1,8,8,111)
        super().__init__(input_sample, config=DeepChessFEConfig(**dict(
            hidden_dims=hidden_dims,
            activations=activations,
            batch_norm=batch_norm,
        )))

class DCEvaluator(DeepChessEvaluator, PyTorchModelHubMixin):
    def __init__(
            self,
            feature_extractor:Type[Model]=None,
            feature_extractor_config:ModelConfig=None,
            feature_extractor_param_dir:Union[str, Path]=None,
            hidden_dims:Union[int, List[int]]=[512, 252, 128],
            activations:Union[str, List[str]]='relu',
            batch_norm:bool=True,) -> None:
        input_sample = torch.rand(1,8,8,111)
        if feature_extractor is None:
            feature_extractor = DCFE.from_pretrained("mzimm003/DeepChessReplicationFeatureExtractor")
        super().__init__(input_sample, config=DeepChessEvaluatorConfig(**dict(
            feature_extractor=feature_extractor,
            feature_extractor_config=feature_extractor_config,
            feature_extractor_param_dir=feature_extractor_param_dir,
            hidden_dims=hidden_dims,
            activations=activations,
            batch_norm=batch_norm,
        )))

class DCMinMax(DeepChessAlphaBeta, PyTorchModelHubMixin):
    def __init__(
            self,
            board_evaluator:Type[Model]=None,
            board_evaluator_config:ModelConfig=None,
            board_evaluator_param_dir:Union[str, Path]=None,
            max_depth:int = 8,
            iterate_depths:bool = True,
            move_sort:Literal['none', 'random', 'evaluation'] = 'evaluation') -> None:
        input_sample = torch.rand(1,8,8,111)
        if board_evaluator is None:
            board_evaluator = DCFE.from_pretrained("mzimm003/DeepChessReplicationBoardEvaluator")
        super().__init__(input_sample, config=DeepChessAlphaBetaConfig(**dict(
            board_evaluator=board_evaluator,
            board_evaluator_config=board_evaluator_config,
            board_evaluator_param_dir=board_evaluator_param_dir,
            max_depth=max_depth,
            iterate_depths=iterate_depths,
            move_sort=move_sort,
        )))

def get_model_attrs(dir:Union[str, Path]):
    best_model_dir = Path(dir).resolve()
    best_model_class = None
    best_model_config = None
    with open(best_model_dir/"params.pkl",'rb') as f:
        x = pickle.load(f)
        best_model_class = x['model']
        best_model_config = x['model_config']

    latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'
    return best_model_class, best_model_config, latest_checkpoint

def main1(kwargs=None):

    mod_cls, mod_config, mod_chkpt = get_model_attrs("/opt/ray/results/ChessFeatureExtractor/AutoEncoder_1557d_00000_0_batch_size=256,model_config=ref_ph_a52f5213,lr=0.0001_2024-07-12_22-30-58")
    model = DCFE(**mod_config.asDict())
    model.load_state_dict(torch.load(mod_chkpt,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.save_pretrained('/opt/pretrained_models/DCFE')
    model.push_to_hub(
        "mzimm003/DeepChessReplicationFeatureExtractor")

    mod_cls, mod_config, mod_chkpt = get_model_attrs("/opt/ray/results/DeepChessEvaluator/ChessEvaluation_9866d_00000_0_learning_rate_scheduler_config=step_size_200_gamma_0_9,model_config=ref_ph_a52f5213,lr=0.1000_2024-07-13_09-18-52")
    model = DCEvaluator(**mod_config.asDict())
    model.load_state_dict(torch.load(mod_chkpt,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.save_pretrained('/opt/pretrained_models/DCEvaluator')
    model.push_to_hub(
        "mzimm003/DeepChessReplicationBoardEvaluator")

    model = DCMinMax(
        **dict(
            board_evaluator=mod_cls,
            board_evaluator_config=mod_config,
            board_evaluator_param_dir=mod_chkpt,
            max_depth=3,
            move_sort='evaluation'))
    model.save_pretrained('/opt/pretrained_models/DCMinMax')
    model.push_to_hub(
        "mzimm003/DeepChessReplicationMinMax")
                      

def main(kwargs=None):
    modelfe = DCFE.from_pretrained("mzimm003/DeepChessReplicationFeatureExtractor")
    modeleval = DCEvaluator.from_pretrained("mzimm003/DeepChessReplicationBoardEvaluator")
    modelminmax = DCMinMax.from_pretrained("mzimm003/DeepChessReplicationMinMax")
    pass

if __name__ == "__main__":
    main()