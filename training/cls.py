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
import gym_symmetric
from torch import nn


def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        print("initial state: ", env.reset())
        print("Observation Space: ", env.observation_space)
        print("Action Space: ", env.action_space)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    layer_n = int(sys.argv[1])
    layer_l = int(sys.argv[2])
    mil_steps = int(sys.argv[3])
    eval_n = int(sys.argv[4])
    eval_m = int(sys.argv[5])
    eval_k = int(sys.argv[6])
    lrate = int(sys.argv[7])

    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env("symmetric-v0", i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    layers = [layer_n for _ in range(layer_l)]
    policy_kwargs = {
        "net_arch": [{"vi": layers, "vf": layers}],
        "activation_fn": nn.ReLU
    }
    network_type = '-'.join(list(map(str, layers)))
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=0,
                gamma=0.99 ** (1 / eval_k), gae_lambda=0.95 ** (1 / eval_k),
                n_steps=256 * eval_k, learning_rate=lrate * 1e-6)

    # model = PPO.load("./data/1mil")
    # model.set_env(env)

    nid = "multi-agent"
    dire = f"./data/n16v500/{network_type}-lrm{lrate}/"

    debug_info = ["reward", "queue", "price", "gain", "operating_cost", "wait_penalty", "overflow", "imitation_reward"]

    for i in range(mil_steps):
        model.learn(total_timesteps=1_000_000)
        model.save(dire + f"{nid}/{i + 1}")
        accu = 0

        lists = {key: [] for key in debug_info}

        for _ in range(eval_n):
            sums = {key: np.zeros((num_cpu,)) for key in debug_info}
            j = 0
            obs = env.reset()
            for _ in range(eval_m * eval_k):
                j += 1
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                if j % eval_k == 0:
                    for k in debug_info:
                        sums[k] += np.array(([v[k] for v in info]))
            for k in debug_info:
                lists[k].extend((sums[k] / eval_m).tolist())

        filename = dire + f"{nid}/stats/reward.tsv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        for k in debug_info:
            with open(dire + f"{nid}/stats/{k}.tsv", 'a') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(lists[k])
        for k in debug_info:
            print(f"{network_type}/{nid}/{i + 1}: {k}: ", statistics.mean(lists[k]))
