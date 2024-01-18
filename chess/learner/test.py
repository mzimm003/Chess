from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy

from pettingzoo.classic import chess_v5
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
import numpy as np

def env_creator(args):
    env = chess_v5.env()
    return env

register_env("Chess",lambda config: PettingZooEnv(env_creator(config)))

if __name__ == "__main__":
    checkpoints = ['/tmp/main_v1', '/tmp/main_v1']
    white_idx = np.random.choice(range(len(checkpoints)),1).item()
    pol1 = Policy.from_checkpoint(checkpoints[white_idx])
    pol2 = Policy.from_checkpoint(checkpoints[1-white_idx])

    # checkpoint = './results/test/PPO_Chess_cef9e_00000_0_lr=0.0025,fcnet_hiddens=1280_2023-08-08_20-35-38/checkpoint_000175/'
    # alg = Algorithm.from_checkpoint(
    #     checkpoint,
    #     policy_ids=['main']
    # )
    # pol = alg.get_policy('main')
    
    env = chess_v5.env('human')

    for i in range(100):
        env.reset()
        epRew = 0
        done = False
        j = 0
        while not done:
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            # for k,v in observation.items():
            #     if isinstance(v, np.ndarray):
            #         observation[k] = v.tolist()
            if not done:
                act = None
                if observation['observation'][:,:,4].all():
                    act = pol2.compute_single_action(obs=observation)[0]
                else:
                    act = pol1.compute_single_action(obs=observation)[0]

                env.step(act)
