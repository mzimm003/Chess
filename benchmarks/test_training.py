import pytest

from pathlib import Path
import tempfile

import ray
from ray import tune
from ray import air

import torch
import os
import shutil

from ml_training_suite.datasets import Dataset
from ml_training_suite.training.base import ray_trainable_wrap
from ml_training_suite.training import (
    TrainingScript,
    Trainer,
    Optimizer,
    Criterion
)
from ml_training_suite.callbacks.base import Callback

from my_chess.dataset import Filters
from my_chess.models import (
    DeepChessEvaluator,
    DeepChessEvaluatorConfig,
)
from my_chess.callback import CustomSupervisedTrainingCallback

BATCHSIZES = [(32),(64),(128),(256),(512),(1024),]
NUMWORKERS = [(1),(2),(4),(5),(6),(8)]

@pytest.mark.parametrize("BATCHSIZE",BATCHSIZES)
def test_training_by_batchsize(BATCHSIZE, benchmark):
    tmpdirname = tempfile.mkdtemp()
    experiment = "TrainableWrapper_dd230_00000_0_2024-10-25_00-41-25"
    model_choice = "<class 'my_chess.models.deepchess.DeepChessFE'>0"
    model_dir = (Path("/opt/ray/results/ChessFeatureExtractor")/experiment).resolve()

    latest_checkpoint_dir = sorted(model_dir.glob('checkpoint*'), reverse=True)[0]
    model_path = latest_checkpoint_dir/model_choice/'model.pt'

    num_trials = 2
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    EPOCHS=6
    cpu_per_trial = num_cpus//num_trials
    gpu_per_trial = num_gpus/num_trials
    dataset_file=Path("/home/user/hdd_datasets/Chess-CCRL-404/database.h5").resolve()
    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        storage=tmpdirname
        )
    def test():
        tuner = tune.Tuner(
            ray_trainable_wrap(
                TrainingScript.SupervisedTraining,
                num_cpus=cpu_per_trial,
                num_gpus=gpu_per_trial),
            run_config=air.RunConfig(
                name="DeepChessEvaluator",
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=2),
                stop={"training_iteration": EPOCHS}),
            param_space=dict(
                dataset=Dataset.ChessDataWinLossPairsSmall,
                dataset_kwargs=dict(
                    dataset_dir=dataset_file,
                    filters=[Filters.draw],
                    static_partners=False,
                    size = 2048*8,
                ),
                pipelines=None,
                trainer_class=Trainer.ChessDataWinLossPairsTrainer,
                autoencoding=False,
                incremental=False,
                models=DeepChessEvaluator,
                models_kwargs=dict(
                    config=DeepChessEvaluatorConfig(
                        feature_extractor_dir = model_path,
                        hidden_dims = [512, 256, 128],
                        activations = 'relu',
                        batch_norm = True,
                    )
                ),
                optimizers=Optimizer.SGD,
                optimizers_kwargs=dict(lr=tune.grid_search([0.05, 0.1])),
                lr_schedulers=None,
                lr_schedulers_kwargs=None,
                criterion=Criterion.CrossEntropy,
                criterion_kwargs=dict(),
                save_path=None,
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=BATCHSIZE,
                shuffle=True,
                num_workers=cpu_per_trial,
                callback=Callback.CustomSupervisedTrainingCallback,
                callback_kwargs=dict(
                    target_label_probability_dim=-1,
                    model_label_probability_dim=-1,
                ),
                save_onnx=False
            )
        )
        tuner.fit()
    
    benchmark.pedantic(test, rounds=3, iterations=5)
    ray.shutdown()
    shutil.rmtree(tmpdirname)
    print("Done")

# @pytest.mark.parametrize("NUMWORKER",NUMWORKERS)
# def test_training_by_num_workers(NUMWORKER, benchmark):
#     tmpdirname = tempfile.mkdtemp()
#     experiment = "TrainableWrapper_dd230_00000_0_2024-10-25_00-41-25"
#     model_choice = "<class 'my_chess.models.deepchess.DeepChessFE'>0"
#     model_dir = (Path("/opt/ray/results/ChessFeatureExtractor")/experiment).resolve()

#     latest_checkpoint_dir = sorted(model_dir.glob('checkpoint*'), reverse=True)[0]
#     model_path = latest_checkpoint_dir/model_choice/'model.pt'

#     num_trials = 2
#     num_cpus = os.cpu_count()
#     num_gpus = torch.cuda.device_count()
#     EPOCHS=6
#     cpu_per_trial = num_cpus//num_trials
#     gpu_per_trial = num_gpus/num_trials
#     dataset_file=Path("/home/user/hdd_datasets/Chess-CCRL-404/database.h5").resolve()
#     def test():
#         assert torch.cuda.is_available()
#         assert num_gpus == 1
#         ray.init(
#             num_cpus=num_cpus,
#             num_gpus=num_gpus,
#             storage=tmpdirname
#             )
#         tuner = tune.Tuner(
#             ray_trainable_wrap(
#                 TrainingScript.SupervisedTraining,
#                 num_cpus=cpu_per_trial,
#                 num_gpus=gpu_per_trial),
#             run_config=air.RunConfig(
#                 name="DeepChessEvaluator",
#                 checkpoint_config=air.CheckpointConfig(checkpoint_frequency=2),
#                 stop={"training_iteration": EPOCHS}),
#             param_space=dict(
#                 dataset=Dataset.ChessDataWinLossPairsSmall,
#                 dataset_kwargs=dict(
#                     dataset_dir=dataset_file,
#                     filters=[Filters.draw],
#                     static_partners=False,
#                     size = 2048*8,
#                 ),
#                 pipelines=None,
#                 trainer_class=Trainer.ChessDataWinLossPairsTrainer,
#                 autoencoding=False,
#                 incremental=False,
#                 models=DeepChessEvaluator,
#                 models_kwargs=dict(
#                     config=DeepChessEvaluatorConfig(
#                         feature_extractor_dir = model_path,
#                         hidden_dims = [512, 256, 128],
#                         activations = 'relu',
#                         batch_norm = True,
#                     )
#                 ),
#                 optimizers=Optimizer.SGD,
#                 optimizers_kwargs=dict(lr=tune.grid_search([0.05, 0.1])),
#                 lr_schedulers=None,
#                 lr_schedulers_kwargs=None,
#                 criterion=Criterion.CrossEntropy,
#                 criterion_kwargs=dict(),
#                 save_path=None,
#                 balance_training_set=False,
#                 k_fold_splits=1,
#                 batch_size=256,
#                 shuffle=True,
#                 num_workers=NUMWORKER,
#                 callback=Callback.CustomSupervisedTrainingCallback,
#                 callback_kwargs=dict(
#                     target_label_probability_dim=-1,
#                     model_label_probability_dim=-1,
#                 ),
#                 save_onnx=False
#             )
#         )
#         tuner.fit()
#         ray.shutdown()
    
#     benchmark.pedantic(test, rounds=3, iterations=5)
#     shutil.rmtree(tmpdirname)
#     print("Done")