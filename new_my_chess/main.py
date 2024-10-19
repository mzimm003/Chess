from ml_training_suite.base import ML_Element
from ml_training_suite.models import Model, ModelConfig, Policy
from ml_training_suite.environments import Environment
from ml_training_suite.training.supervised import SupervisedTraining

from my_chess.models import DeepChessAlphaBeta, DeepChessAlphaBetaConfig
from my_chess.dataset import ChessDataGenerator
from my_chess.environment import Chess

from quickscript.scripts import Script, ScriptChooser

import os
import pickle
from pathlib import Path
from typing import (
    List,
    Union,
    Type,
    Callable,
    TypeVar
)
from functools import partial

import pygame
from pettingzoo.classic.chess import chess_utils

import torch
import streamlit as st
import numpy as np

from ray import tune, air

T = TypeVar("T", bound=ML_Element)
def ensure_initialization(
        obj:Union[str, ML_Element, Type[ML_Element]],
        typ:Type[T]) -> T:
    return (obj
            if isinstance(obj, typ)
            else typ.initialize(obj))

class HumanVsBot(Script):
    class HumanInputHandler:
        def __init__(
                self,
                ui:str,
                ):
            self.ui = ui
            self.player_num = None

        def set_player_num(self, num):
            self.player_num = num

        def get_player_num(self):
            return self.player_num

        def window_size(self):
            res = None
            if self.ui == "pygame":
                res = pygame.display.get_surface().get_size()
            elif self.ui == "streamlit":
                res = (st.session_state["board"]["width"], st.session_state["board"]["height"])
            return res

        def pos_to_square(self, x, y):
            window_size = self.window_size()
            square_width = window_size[0] // 8
            square_height = window_size[1] // 8
            return int(x // square_width), int((window_size[1] - y) // square_height)
        
        def square_to_coord(self, x, y):
            return ''.join((
                chr(ord('a') + x),
                str(y + 1)
                ))
        
        def squares_to_move(self, f, t):
            return self.square_to_coord(*f) + self.square_to_coord(*t)

        def get(self, observation, env, **kwargs):
            action = None
            from_coord = None
            legal_actions = chess_utils.legal_moves(env.board)
            legal_moves = [str(chess_utils.action_to_move(env.board, x, self.player_num)) for x in legal_actions]
            legally_moved = False
            while not legally_moved:
                if self.ui == "pygame":
                    ev = pygame.event.get()
                    for event in ev:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            x, y = event.pos
                            from_coord = self.pos_to_square(x, y)
                        if event.type == pygame.MOUSEBUTTONUP:
                            x, y = event.pos
                            to_coord = self.pos_to_square(x, y)
                            if to_coord == from_coord:
                                # clicked to pick piece
                                attempted_move = False
                                while not attempted_move:
                                    ev = pygame.event.get()
                                    for event in ev:
                                        if event.type == pygame.MOUSEBUTTONDOWN:
                                            x, y = event.pos
                                            to_coord = self.pos_to_square(x, y)
                                        if event.type == pygame.MOUSEBUTTONUP:
                                            x, y = event.pos
                                            if self.pos_to_square(x, y) == to_coord:
                                                attempted_move = True
                                                move = self.squares_to_move(from_coord, to_coord)
                                                if not move in legal_moves:
                                                    move = move + 'q' #hack to incorporate promoting to queen
                                                if move in legal_moves:
                                                    action = legal_actions[legal_moves.index(move)]
                                                    legally_moved = True
                            else:
                                # dragged piece
                                move = self.squares_to_move(from_coord, to_coord)
                                if not move in legal_moves:
                                    move = move + 'q' #hack to incorporate promoting to queen
                                if move in legal_moves:
                                    action = legal_actions[legal_moves.index(move)]
                                    legally_moved = True
                else:
                    board_return = st.session_state["board"]
                    if not board_return is None:
                        x, y = (board_return["x1"], board_return["y1"])
                        from_coord = self.pos_to_square(x, y)
                        x, y = (board_return["x2"], board_return["y2"])
                        to_coord = self.pos_to_square(x, y)

                        if from_coord != to_coord:
                            # mouse was dragged
                            move = self.squares_to_move(from_coord, to_coord)
                            if not move in legal_moves:
                                move = move + 'q' #hack to incorporate promoting to queen
                            if move in legal_moves:
                                action = legal_actions[legal_moves.index(move)]
                                legally_moved = True
                        else:
                            # piece was clicked
                            pass
            return action

    def __init__(
            self,
            checkpoint:List[Union[str,Path]]=None,
            model:Union[Type[Model],List[Type[Model]]]=None,
            model_config:Union[ModelConfig,List[ModelConfig]]=None,
            environment:Union[Environment, str]=None,
            extra_model_environment_context:Callable=None,
            **kwargs) -> None:
        """
        Args:
            checkpoint: Directory location of policy to visualize.
            model: Extra means of creating a policy. Will overwrite checkpoint
              input.
            model_config: Configuration parameters of model, must be supplied if
              model is.
            environment: Name of environment in which to play.
            extra_model_environment_context: function which operates on
              environment to provide additional input to model.
        """
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.environment = ensure_initialization(environment, Environment)
        self.policies = None
        if self.checkpoint:
            if not isinstance(self.checkpoint, list):
                self.checkpoint = [self.checkpoint]
            self.policies = [Policy.from_checkpoint(chkpt) for chkpt in self.checkpoint]
        self.model = model
        self.model_config = model_config
        self.ui = "pygame" if environment.render_mode == "human" else "streamlit"
        self.done = None
        input_sample, _ = self.environment.reset()
        self.human_input = HumanVsBot.HumanInputHandler(
            ui = self.ui)
        input_sample = next(iter(input_sample.values()))
        self.extra_model_environment_context = extra_model_environment_context if extra_model_environment_context else lambda x: {}
        if self.model:
            if not isinstance(self.model, list):
                self.model = [self.model]
                self.model_config = [self.model_config]
            self.policies = [
                m(input_sample=input_sample, config=m_c)
                if isinstance(m, Type) else
                m
                for m, m_c in zip(self.model, self.model_config)]
            for p in self.policies:
                p.eval()
        
        assert len(self.policies) == len(self.environment.agents) - 1
        pols = [partial(self.get_ai_input, model=mod) for mod in self.policies]
        pols.append(self.human_input.get)
        agent_assignment = np.random.choice(len(pols), len(self.environment.agents), replace=False)
        self.action_map = {}
        for i, j in enumerate(agent_assignment):
            self.action_map[self.environment.agents[i]] = pols[j]
            if pols[j] == self.human_input.get:
                self.human_input.set_player_num(i)

    def get_human_player(self):
        return self.environment.agents[self.human_input.get_player_num()]

    def get_curr_player(self):
        return self.environment.env.agent_selection
    
    def get_result(self):
        return self.environment.env.board.result(claim_draw=True)
    
    def is_done(self):
        return self.done

    def square_num(self, x, y):
        return y * 8 + x

    def get_ai_input(self, observation, env, model:Union[Policy, Model]):
        act = None
        if isinstance(model, Policy):
            act, _ = model.compute_single_action(obs=observation)
        else:
            act = model(input=observation, **self.extra_model_environment_context(env))
        return act
    
    def render_board(self):
        return self.environment.render()

    def run(self):
        """
        Args:
        """
        self.done = False
        while not self.done:
            observation, reward, termination, truncation, info = self.environment.env.last()
            self.done = termination or truncation
            if not self.done:
                actor = self.environment.env.agent_selection
                act = self.action_map[actor](observation=observation, env=self.environment.env)
                self.environment.env.step(act)
            if self.ui == "streamlit":
                st.rerun()
        
class Premade(Script):
    def __init__(
            self,
            debug:bool=False,
            script:str="train_class_ray",
            **kwargs) -> None:
        """
        Args:
            debug: Flag to run script for debugging.
            script: Premade script configuration to be run.
        """
        super().__init__(**kwargs)
        self.debug = debug
        self.script = script

    def createDataset(self):
        dbg = ChessDataGenerator(
            dir=Path("/home/user/hdd_datasets/Chess-CCRL-404"),
            init_size=2**24,
            chunk_size=0,
            max_size=None,
            resize_step=2**23,
            gen_batch_size=512,
            compression='gzip',
            compression_opts=4,
            oracle=True,
            oracle_path="/home/user/Programming/stockfish/src/stockfish",
            oracle_limit_config={"time":.05},
            exclude_draws=False,
            exclude_x_initial_moves=6,
            reset=False,
            states_per_game=20,
            subset=None,
            seed=None,
            debug=self.debug,)
        dbg.create_database()

    def playVsAI(self):
        # best_model_dir = Path("./results/DeepChessEvaluator/ChessEvaluation_d8721_00002_2_batch_size=256,learning_rate_scheduler_config=milestones___ref_ph_33c73b19_gamma_0_75,model_config=r_2024-03-13_10-19-29").resolve()
        best_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ChessEvaluation_9866d_00000_0_learning_rate_scheduler_config=step_size_200_gamma_0_9,model_config=ref_ph_a52f5213,lr=0.1000_2024-07-13_09-18-52").resolve()
        # best_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ModelDistill_1a05b_00000_0_batch_size=256,learning_rate=0.0100,learning_rate_scheduler_config=step_size_1_gamma_0_95,model_config=_2024-03-29_10-45-15").resolve()
        
        best_model_class = None
        best_model_config = None
        with open(best_model_dir/"params.pkl",'rb') as f:
            x = pickle.load(f)
            best_model_class = x['model']
            best_model_config = x['model_config']
            # best_model_config.feature_extractor_param_dir = Path(str(best_model_config.feature_extractor_param_dir).replace("/opt","/home/mark"))

        latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'
        play = HumanVsBot(
            model=DeepChessAlphaBeta,
            model_config=DeepChessAlphaBetaConfig(
                board_evaluator=best_model_class,
                board_evaluator_config=best_model_config,
                board_evaluator_param_dir=latest_checkpoint,
                max_depth=8,
                move_sort='evaluation',
                iterate_depths=True
            ),
            environment=Chess(render_mode="human"),
            extra_model_environment_context=lambda env: {"board":env.board}
        )
        play.run()

    def trainFeatureExtractor(self):
        num_trials = 1
        num_cpus = 1 if self.debug else os.cpu_count()
        num_gpus = 0 if self.debug else torch.cuda.device_count()
        BATCHSIZE=256
        EPOCHS=20
        cpu_per_trial = num_cpus//num_trials
        gpu_per_trial = num_gpus/num_trials
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            None,
            ]
        save_path=Path("./models/classifier").resolve()
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            local_mode=self.debug,
            storage="/opt/ray/results"
            )
        train_script = Train(
            # debug=True,
            num_cpus=16,
            num_gpus=0.85,
            training_on="ChessData",
            algorithm="AutoEncoder",
            algorithm_config=AutoEncoderConfig(
                learning_rate = tune.grid_search([0.0001]),
                # learning_rate = tune.grid_search([0.0002]),
                # learning_rate = tune.grid_search([0.000025]),
                batch_size = tune.grid_search([256]),
                # batch_size = tune.grid_search([256, 512, 2048, 4096])
                # learning_rate_scheduler=torch.optim.lr_scheduler.StepLR,
                # learning_rate_scheduler_config=tune.grid_search([dict(step_size=25, gamma=0.2)]),
                criterion=CrossEntropyLoss,
                criterion_config={"class_index":-1},
                data_split=[.8,.1,.1]
            ),
            # model="TransformerChessFE",
            model="DeepChessFE",
            model_config=tune.grid_search([
                # DeepChessFEConfig(hidden_dims=[4096, 2048, 1024, 512, 256, 128]),
                DeepChessFEConfig(hidden_dims=[4096, 2048, 512, 128]),
                # DeepChessFEConfig(hidden_dims=[4096, 1024, 256, 128]),
                # DeepChessFEConfig(hidden_dims=[4096, 1024, 512, 128]),
                # DeepChessFEConfig(hidden_dims=[4096, 1024, 128]),
                # DeepChessFEConfig(hidden_dims=[4096, 512, 128])
                # TransformerChessFEConfig(pad_features=True)
                ]),
            # model_config="DeepChessFEConfig",
            run_config=air.RunConfig(
                                name="ChessFeatureExtractor",
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
                                # stop={"timesteps_total": 20},
                                stop={"training_iteration": 20},
                                ),
        )
        # (train_script.getAlgConfig()
        #     .multi_agent(
        #             # policy_mapping_fn=policy_mapping_fn,
        #             policy_map_capacity=2,
        #             )
        #     .callbacks(SelfPlayCallback)
        #     .training(
        #         lr_schedule=tune.grid_search([[[1,0.00001],[7000000,0.000001]],
        #                                       [[1,0.00005],[7000000,0.000005],[9000000,0.000001]],
        #                                       [[1,0.000025],[7000000,0.0000025],[9000000,0.000001]]]),
        #         # optimizer={'simple_optimizer':True},
        #         # kl_coeff=.005,
        #     )
        #     .resources(num_gpus=0.85)
        #     .framework(framework='torch')
        #     .rollouts(num_rollout_workers=11, num_envs_per_worker=50)
        # )
        train_script.run()

    def trainFeatureExtractorDistillation(self):
        best_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ChessEvaluation_9866d_00000_0_learning_rate_scheduler_config=step_size_200_gamma_0_9,model_config=ref_ph_a52f5213,lr=0.1000_2024-07-13_09-18-52").resolve()
        best_model_class = None
        best_model_config = None
        x = None
        with open(best_model_dir/"params.pkl",'rb') as f:
            x = pickle.load(f)
            best_model_class = x['model']
            best_model_config = x['model_config']

        latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

        temp_model = best_model_class(input_sample=torch.rand((2,8,8,111)) ,config=best_model_config)
        temp_model.load_state_dict(torch.load(latest_checkpoint))

        teacher_model = x["model_config"].feature_extractor
        teacher_model_config = x["model_config"].feature_extractor_config
        temp_teacher_model_param_path = Path("/tmp/tmp_model.pt")
        torch.save(temp_model.fe.state_dict(), temp_teacher_model_param_path)

        train_script = Train(
            # debug=True,
            num_cpus=8,
            num_gpus=0.45,
            system_cpus=16,
            system_gpus=1,
            training_on="ChessData",
            algorithm="ModelDistill",
            algorithm_config=ModelDistillConfig(
                dataset_config=dict(dataset_dir='/home/user/ssd_datasets/Chess-CCRL-404'),
                batch_size = tune.grid_search([256]),
                # optimizer=torch.optim.SGD,
                optimizer=torch.optim.Adam,
                learning_rate = tune.grid_search([0.001, 0.0005]),
                # learning_rate = 0.001,
                # learning_rate = tune.grid_search([0.0001, 0.00005]),
                learning_rate_scheduler=torch.optim.lr_scheduler.StepLR,
                learning_rate_scheduler_config=tune.grid_search([
                    dict(step_size=200, gamma=0.9),
                    # dict(step_size=1, gamma=0.99),
                    # dict(step_size=1, gamma=0.95),
                    # dict(step_size=1, gamma=0.9),
                    # dict(step_size=1, gamma=0.85),
                    # dict(step_size=1, gamma=0.8),
                    # dict(step_size=1, gamma=0.75),
                    # dict(milestones=range(1,20), gamma=0.75),
                    # dict(milestones=range(1,25), gamma=0.75),
                    # dict(milestones=range(1,30), gamma=0.75),
                    ]),
                parent_model=teacher_model,
                parent_model_config=teacher_model_config,
                parent_model_param_dir=temp_teacher_model_param_path,
                train_on_teacher_only=tune.grid_search([True])
            ),
            model="DeepChessFE",
            model_config=tune.grid_search([
                DeepChessFEConfig(
                    hidden_dims=[128, 128, 128],
                    batch_norm=True),
                ]),
            run_config=air.RunConfig(
                                name="ChessFeatureExtractor",
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5),
                                stop={"training_iteration": 20},
                                storage_path="/opt/ray/results"
                                ),
        )
        train_script.run()

    def trainEvaluator(self):
        best_model_dir = Path("/opt/ray/results/ChessFeatureExtractor/AutoEncoder_1557d_00000_0_batch_size=256,model_config=ref_ph_a52f5213,lr=0.0001_2024-07-12_22-30-58").resolve()
        # best_model_dir = Path("./results/ChessFeatureExtractor/AutoEncoder_8e326_00004_4_batch_size=256,model_config=ref_ph_d2c2b490,lr=0.0001_2024-02-27_15-47-24").resolve()
        best_model_class = None
        best_model_config = None
        with open(best_model_dir/"params.pkl",'rb') as f:
            x = pickle.load(f)
            best_model_class = x['model']
            best_model_config = x['model_config']

        latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

        train_script = Train(
            # debug=True,
            num_cpus=16,
            num_gpus=0.85,
            # restore='/opt/ray/results/DeepChessEvaluator',
            training_on="ChessDataWinLossPairs",
            algorithm="ChessEvaluation",
            algorithm_config=ChessEvaluationConfig(
                dataset_config=dict(dataset_dir='/opt/datasets/Chess-CCRL-404', static_partners=False),
                # batch_size = tune.grid_search([16384]),
                optimizer=torch.optim.SGD,
                # optimizer=torch.optim.Adam,
                # learning_rate = tune.grid_search([0.1]),
                learning_rate = tune.grid_search([0.1,0.001,0.0001]),
                learning_rate_scheduler=torch.optim.lr_scheduler.StepLR,
                learning_rate_scheduler_config=tune.grid_search([
                    dict(step_size=200, gamma=0.9),
                    # dict(step_size=1, gamma=0.99),
                    # dict(step_size=1, gamma=0.95),
                    # dict(step_size=1, gamma=0.9),
                    # dict(step_size=1, gamma=0.85),
                    # dict(step_size=1, gamma=0.8),
                    # dict(step_size=1, gamma=0.75),
                    # dict(milestones=range(1,20), gamma=0.75),
                    # dict(milestones=range(1,25), gamma=0.75),
                    # dict(milestones=range(1,30), gamma=0.75),
                    ]),
                data_split=(.8,.1,.1)
            ),
            model="DeepChessEvaluator",
            model_config=tune.grid_search([
                DeepChessEvaluatorConfig(
                    feature_extractor=best_model_class,
                    feature_extractor_config=best_model_config,
                    feature_extractor_param_dir=latest_checkpoint,
                    hidden_dims=[512, 256, 128],
                    batch_norm=True),
                ]),
            run_config=air.RunConfig(
                                name="DeepChessEvaluator",
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5),
                                stop={"training_iteration": 20},
                                ),
        )
        train_script.run()

    def trainEvaluatorDistillation(self):
        teacher_model_dir = Path("/opt/ray/results/DeepChessEvaluator/ChessEvaluation_9866d_00000_0_learning_rate_scheduler_config=step_size_200_gamma_0_9,model_config=ref_ph_a52f5213,lr=0.1000_2024-07-13_09-18-52").resolve()
        teacher_model_class = None
        teacher_model_config = None
        x = None
        with open(teacher_model_dir/"params.pkl",'rb') as f:
            x = pickle.load(f)
            teacher_model_class = x['model']
            teacher_model_config = x['model_config']

        teacher_model_param_path = sorted(teacher_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

        fe_model_dir = Path("/opt/ray/results/ChessFeatureExtractor/ModelDistill_19f52_00000_0_batch_size=256,learning_rate=0.0100,learning_rate_scheduler_config=step_size_1_gamma_0_95,model_config=_2024-03-29_01-41-14").resolve()
        fe_model_class = None
        fe_model_config = None
        x = None
        with open(fe_model_dir/"params.pkl",'rb') as f:
            x = pickle.load(f)
            fe_model_class = x['model']
            fe_model_config = x['model_config']

        fe_model_param_path = sorted(fe_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

        train_script = Train(
            # debug=True,
            num_cpus=16,
            num_gpus=0.85,
            training_on="ChessDataWinLossPairs",
            algorithm="ModelDistill",
            algorithm_config=ModelDistillConfig(
                dataset_config=dict(dataset_dir='/opt/datasets/Chess-CCRL-404', static_partners=False),
                batch_size = tune.grid_search([256]),
                optimizer=torch.optim.SGD,
                # optimizer=torch.optim.Adam,
                learning_rate = tune.grid_search([0.01]),
                # learning_rate = tune.grid_search([0.0001, 0.00005]),
                learning_rate_scheduler=torch.optim.lr_scheduler.StepLR,
                learning_rate_scheduler_config=tune.grid_search([
                    # dict(step_size=200, gamma=0.9),
                    # dict(step_size=1, gamma=0.99),
                    dict(step_size=1, gamma=0.95),
                    # dict(step_size=1, gamma=0.9),
                    # dict(step_size=1, gamma=0.85),
                    # dict(step_size=1, gamma=0.8),
                    # dict(step_size=1, gamma=0.75),
                    # dict(milestones=range(1,20), gamma=0.75),
                    # dict(milestones=range(1,25), gamma=0.75),
                    # dict(milestones=range(1,30), gamma=0.75),
                    ]),
                parent_model=teacher_model_class,
                parent_model_config=teacher_model_config,
                parent_model_param_dir=teacher_model_param_path,
            ),
            model="DeepChessEvaluator",
            model_config=tune.grid_search([
                DeepChessEvaluatorConfig(
                    feature_extractor=fe_model_class,
                    feature_extractor_config=fe_model_config,
                    feature_extractor_param_dir=fe_model_param_path,
                    hidden_dims=[128, 128],
                    batch_norm=False),
                ]),
            run_config=air.RunConfig(
                                name="DeepChessEvaluator",
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
                                stop={"training_iteration": 100},
                                ),
        )
        train_script.run()

    def trainRL(self):
        best_model_dir = Path("./results/ChessFeatureExtractor/AutoEncoder_5a829_00000_0_batch_size=256,model_config=ref_ph_a52f5213,lr=0.0001_2024-03-07_00-47-39").resolve()
        best_model_class = None
        best_model_config = None
        with open(best_model_dir/"params.pkl",'rb') as f:
            x = pickle.load(f)
            best_model_class = x['model']
            best_model_config = x['model_config']

        latest_checkpoint = sorted(best_model_dir.glob('checkpoint*'), reverse=True)[0]/'model.pt'

        train_script = Train(
            debug=True,
            num_cpus=16,
            num_gpus=0.95,
            training_on="Chess",
            algorithm="PPO",
            algorithm_config="PPOConfig",
            policy="PPOPolicy",
            policy_config="PPOPolicyConfig",
            model="DeepChessRL",
            model_config=DeepChessRLConfig(
                feature_extractor=best_model_class,
                feature_extractor_config=best_model_config,
                feature_extractor_param_dir=latest_checkpoint,
            ),
            run_config=air.RunConfig(
                                name="DeepChessRL",
                                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10)),
        )
        (train_script.getAlgConfig()
            .multi_agent(
                    # policy_mapping_fn=policy_mapping_fn,
                    policy_map_capacity=2,
                    )
            .callbacks(SelfPlayCallback)
            .training(
                lr=tune.grid_search([0.00001]),
                optimizer={'simple_optimizer':True},
                # kl_coeff=.005,
            )
            .resources(num_gpus=0.95)
            .framework(framework='torch')
            .rollouts(num_rollout_workers=1, num_envs_per_worker=5)
        )
        train_script.run()

    def setup(self):
        premades = {
            "create_dataset":self.createDataset,
            "play_v_ai":self.playVsAI,
            "train_feat_ext":self.trainFeatureExtractor,
            "train_feat_ext_dist":self.trainFeatureExtractorDistillation,
            "train_eval":self.trainEvaluator,
            "train_eval_dist":self.trainEvaluatorDistillation,
            "train_rl":self.trainRL,
        }
        self.script = premades[self.script]

    def run(self):
        """
        """
        self.script()

if __name__ == "__main__":
    ScriptChooser().complete_run()