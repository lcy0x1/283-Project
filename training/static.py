import math
import random
import csv

from gym_symmetric.envs.symmetric_env import VehicleEnv, Imitated

if __name__ == "__main__":
    env = VehicleEnv()
    env.reset()
    env.seed(random.randint(0, 1000000))
    act_vehs = []
    act_pris = []
    filename = 'opt'
    with open(f"../static/{filename}_vehicle.csv", 'r') as in_file:
        reader = csv.reader(in_file, delimiter=',')
        for line in reader:
            act_vehs.append(line)
    with open(f"../static/{filename}_prices.csv", 'r') as in_file:
        reader = csv.reader(in_file, delimiter=',')
        for line in reader:
            act_pris.append(line)
    action = [[] for i in range(env.node)]
    for i in range(env.node):
        action[i].extend([float(act_vehs[i][j]) / 10 for j in range(env.node)])
        action[i].extend([float(act_pris[i][j])+0.25 for j in range(env.node)])

    n = 1000
    cumulative_reward = 0
    cumulative_square = 0
    for cycle in range(n):
        _, reward, _, info = env.cycle_step(action)
        reward = reward
        cumulative_reward += reward
        cumulative_square += reward ** 2

    mean = cumulative_reward / n
    var = cumulative_square / n - mean ** 2
    print("mean: ", mean, "stdev: ", math.sqrt(var / n))
