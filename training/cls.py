import csv
import os
import statistics
import sys

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from network_imitate import dummy_env, ImitateACP


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
    mil_steps = int(sys.argv[1])
    eval_n = int(sys.argv[2])
    eval_m = int(sys.argv[3])
    lrate = int(sys.argv[4])

    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env("route-v1", i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    model = PPO(ImitateACP, env, verbose=0,
                gamma=0.99, gae_lambda=0.95,
                n_steps=256, learning_rate=lrate * 1e-6, device='cuda:1')

    # model = PPO.load("./data/1mil")
    # model.set_env(env)

    nid = "imitate-agent"
    vfn_middle = dummy_env.config["vfn_middle"]
    vfn_m = dummy_env.config["vfn_out"]
    dire = f"/mldata/chengyilin/n49v4000skew08/sp1-{vfn_middle}-{vfn_m}-lrm{lrate}/"

    debug_info = ["reward", "queue", "price", "gain", "operating_cost", "wait_penalty", "overflow", "imitation_reward",
                  "rebalancing_cost", "distance_served"]

    do_reset = False

    for i in range(mil_steps):
        if do_reset:
            for _ in range(10):
                acp: ImitateACP = model.policy
                acp.re_init()
                model.learn(total_timesteps=100_000)
        else:
            model.learn(total_timesteps=1_000_000)
        model.save(dire + f"{nid}/{i + 1}")
        accu = 0

        lists = {key: [] for key in debug_info}

        for _ in range(eval_n):
            sums = {key: np.zeros((num_cpu,)) for key in debug_info}
            j = 0
            obs = env.reset()
            for _ in range(eval_m):
                j += 1
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                for k in debug_info:
                    sums[k] += np.array(([v[k] for v in info]))
            for k in debug_info:
                lists[k].extend((sums[k] / eval_m).tolist())
        if statistics.mean(lists["reward"]) > 0:
            do_reset = False
        filename = dire + f"{nid}/stats/reward.tsv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        for k in debug_info:
            with open(dire + f"{nid}/stats/{k}.tsv", 'a') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(lists[k])
        for k in debug_info:
            print(f"{nid}/{i + 1}: {k}: ", statistics.mean(lists[k]))
