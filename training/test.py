import csv
import json
import os
import statistics
import sys

import gym
import numpy as np
import pkg_resources
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import gym_route
from torch import nn

from training.networks.imitate import ImitateACP
from training.networks.simple import SimpleACP


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    mil_steps = 100
    eval_n = 30
    eval_m = 30
    eval_k = 1
    lrate = 100

    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env("route-v1", i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    model = PPO(ImitateACP, env, verbose=0,
                gamma=0.99 ** (1 / eval_k), gae_lambda=0.95 ** (1 / eval_k),
                n_steps=256 * eval_k, learning_rate=lrate * 1e-6)

    # model = PPO.load("./data/1mil")
    # model.set_env(env)

    nid = "multi-agent"

    debug_info = ["reward", "queue", "price", "gain", "operating_cost", "wait_penalty", "overflow", "imitation_reward"]

    for i in range(mil_steps):
        # model.learn(total_timesteps=1_0_000)
        accu = 0

        lists = {key: [] for key in debug_info}

        for _ in range(eval_n):
            sums = {key: np.zeros((num_cpu,)) for key in debug_info}
            j = 0
            obs = env.reset()
            for _ in range(eval_m * eval_k):
                j += 1
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                if j % eval_k == 0:
                    for k in debug_info:
                        sums[k] += np.array(([v[k] for v in info]))
            for k in debug_info:
                lists[k].extend((sums[k] / eval_m).tolist())

        for k in debug_info:
            print(f"{nid}/{i + 1}: {k}: ", statistics.mean(lists[k]))
