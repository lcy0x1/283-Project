import math
import random
from gym_symmetric.envs.symmetric_env import VehicleEnv, Imitated

env = VehicleEnv()
env.reset()
env.seed(random.randint(0, 1000000))
imitate = Imitated(env)

n = 300
cumulative_reward = 0
cumulative_square = 0
average_price = 0
for cycle in range(n):
    action = imitate.compute_action()
    _, reward, _, info = env.cycle_step(action)
    reward = info["reward"]
    cumulative_reward += reward
    cumulative_square += reward ** 2
    average_price += info["price"]

mean = cumulative_reward / n
var = cumulative_square / n - mean ** 2
print("mean: ", mean, "stdev: ", math.sqrt(var / n))
print("average price", average_price / n)
