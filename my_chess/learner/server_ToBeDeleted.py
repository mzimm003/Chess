#TODO Incorporate into "Scripts"

from pettingzoo.classic import chess_v5
from starlette.requests import Request
from ray.rllib.algorithms.ppo import PPOConfig
from ray import serve
import requests
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv


def env_creator(args):
    env = chess_v5.env()
    return env

register_env("Chess",lambda config: PettingZooEnv(env_creator(config)))

@serve.deployment
class ServePPOModel:
    def __init__(self, checkpoint_path) -> None:
        # Re-create the originally used config.
        config = (PPOConfig()
                    .framework(framework='torch')
                    .rollouts(num_rollout_workers=0)
                    .training(model={'dim':8, 'conv_filters':[[20,[1,1],1]]}))
        self.algorithm = config.build("Chess")
        # Restore the algo's state from the checkpoint.
        self.algorithm.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        # obs = json_input["observation"]

        action = self.algorithm.compute_single_action(json_input)
        return {"action": int(action)}
    
if __name__ == "__main__":
    ppo_model = ServePPOModel.bind('./results/test/PPO_Chess_7fb5d_00000_0_2023-05-07_18-27-30/checkpoint_000075')
    serve.run(ppo_model)

    env = chess_v5.env('human')

    for i in range(100):
        env.reset()
        epRew = 0
        done = False
        j = 0
        while not done:
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            for k,v in observation.items():
                if isinstance(v, np.ndarray):
                    observation[k] = v.tolist()
            if not done:
                resp = requests.get(
                    "http://localhost:8000/", json=observation
                )

                env.step(resp.json()['action'])
