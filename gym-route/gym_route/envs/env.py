from typing import List, Optional

import gym
import math
import pkg_resources
from gym import spaces
from gym.utils import seeding
import json
import sys
import numpy as np

from gym_route.envs.imitate import Imitated
from numpy.random.mtrand import RandomState


class VehicleAction:

    def __init__(self, env, i, arr):
        self.motion = [0 for _ in range(env.node)]
        self.price = [0.0 for _ in range(env.node)]
        ind = 0
        tmp = [0 for _ in range(env.node)]
        rsum = 0
        for j in range(env.node):
            tmp[j] = min(1, max(0, arr[ind]))
            rsum = rsum + tmp[j]
            ind = ind + 1
        rsum = max(1e-5, rsum)
        rem: int = env.vehicles[i]
        for j in range(env.node):
            tmp[j] = env.vehicles[i] * tmp[j] / rsum
            rem = rem - math.floor(tmp[j])
        random = env.random.rand(1)
        rem = rem - 1
        for j in range(env.node):
            mrem = tmp[j] - math.floor(tmp[j])
            if (random > 0) and (random < mrem):
                self.motion[j] = math.floor(tmp[j]) + 1
                if rem > 0:
                    random = random + env.random.rand(1)
                    rem = rem - 1
            else:
                self.motion[j] = math.floor(tmp[j])
            random = random - mrem
        for j in range(env.node):
            if i != j:
                self.price[j] = min(1, max(0, arr[ind]))
            ind = ind + 1


class AverageQueue:

    def __init__(self, cap):
        self.list = [0 for _ in range(cap)]
        self.cap = cap
        self.count = 0
        self.index = 0
        self.sum = 0

    def add(self, val):
        self.sum = self.sum - self.list[self.index] + val
        self.list[self.index] = val
        self.count = min(self.cap, self.count + 1)
        self.index = (self.index + 1) % self.cap

    def average(self):
        if self.count == 0:
            return 0
        return self.sum / self.count

    def reset(self):
        self.list = [0 for _ in range(self.cap)]
        self.count = 0
        self.index = 0
        self.sum = 0


class VehicleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None, seed=0, imitate=True):
        if config is None:
            config = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
        self.config = config
        self.node = self.config["node"]
        self.vehicle = self.config["vehicle"]
        self.poisson_param = self.config["poisson_param"]
        self.operating_cost = self.config["operation_cost"]
        self.waiting_penalty = self.config["waiting_penalty"]
        self.queue_size = self.config["max_queue"]
        self.overflow = self.config["overflow"]
        self.poisson_cap = self.config["poisson_cap"]
        self.use_average_reward = self.config["average_reward"]
        self.average_queue_length = self.config["average_queue"]
        self.average_queue_reset = self.config["average_reset"]
        self.include_edge = self.config["include_edge"]
        self.mini_node_layer = self.config["mini_node_layer"]
        self.multi_agent = self.config["multi_agent"]
        self.imitate = imitate and self.config["imitate"]
        self.reward_factor = self.config["reward_factor"]
        self.skip_second_gradient = self.config["skip_second_gradient"]

        self.vehicles = [0 for _ in range(self.node)]
        self.queue = [[0 for _ in range(self.node)] for _ in range(self.node)]

        # Attempt at edge initialization
        # Edge matrix: self.edge(0) = 1->2 , self.edge(1) = 2->1     for 2 node case (2 edges)
        # n nodes: self.edge(0) = 1->2 , 1->3 , ... 1->n , 2->1 , 2->3, ... 2->n , ... n->n-2 , n->n-1  (? edges)
        self.edge_matrix = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.demand_factor = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.price_factor = [[0 for _ in range(self.node)] for _ in range(self.node)]
        self.bounds = [0 for _ in range(self.node)]
        self.custom_edge = self.config["custom_edge"]
        self.generate_skew = self.config["generate_skew"]
        self.custom_skew = self.config["custom_skew"]
        if self.custom_edge:
            self.edge_list = self.config["edge_lengths"]
            self.demand_list = self.config["demand_factor"]
            self.price_list = self.config["price_factor"]
            self.fill_edge_matrix()
        else:
            self.create_matrix()
        self.max_edge = max(self.bounds)

        self.current_index = 0
        self.action_cache: List[Optional[VehicleAction]] = [None for _ in range(self.node)]
        self.over = 0
        self.random: Optional[RandomState] = None

        if self.multi_agent:
            self.observation_space = spaces.Box(0, np.array(
                [self.vehicle + 1 for _ in range(self.node)] +  # vehicles
                [self.vehicle + 1 for _ in range(self.node * self.mini_node_layer)] +  # vehicles
                [self.queue_size + 1 for _ in range(self.node)] +  # queue
                [self.queue_size * (self.node - 1) + 1 for _ in range(self.node)] +  # queue at other nodes
                ([self.max_edge + 1 for _ in range(self.node)] if self.include_edge else []) +
                [self.node]))  # state
            self.action_space = spaces.Box(0, 1, (self.node * 2,))
        else:
            self.observation_space = spaces.Box(0, np.array(
                [self.vehicle + 1 for _ in range(self.node)] +  # vehicles
                [self.vehicle + 1 for _ in range(self.node * self.mini_node_layer)] +  # mininode
                [self.queue_size + 1 for _ in range(self.node ** 2)]))  # queue
            self.action_space = spaces.Box(0, 1, (self.node ** 2 * 2,))

        # Stores number of vehicles at mini node between i and j
        self.mini_vehicles = [[[0 for _ in range(self.edge_matrix[i][j] - 1)]
                               for j in range(self.node)] for i in range(self.node)]

        self.mini_node_processor = lambda sums, i, j: sums[i]

        self.average_reward = AverageQueue(self.average_queue_length)
        self.seed(seed)

        if self.imitate:
            self.imitation = Imitated(self, VehicleEnv(config, imitate=False))

    def seed(self, seed=None):
        self.random, _ = seeding.np_random(seed)

    def fill_edge_matrix(self):
        edge_num = len(self.edge_list)
        if (self.node * self.node) != edge_num:
            print("Incorrect edge_lengths parameter. Total nodes and edges do not match!")
            sys.exit()
        # Creating 2D matrix for easier access
        tmp = 0
        for i in range(self.node):
            for j in range(self.node):
                self.edge_matrix[i][j] = self.edge_list[tmp]
                self.price_factor[i][j] = self.price_list[tmp]
                self.demand_factor[i][j] = self.demand_list[tmp]
                self.bounds[j] = max(self.bounds[j], self.edge_matrix[i][j])
                tmp += 1
                if i == j:
                    continue
                if self.edge_matrix[i][j] < 1:
                    print("Error! Edge length too short (minimum length 1).")
                    sys.exit()
                if self.edge_matrix[i][j] % 1 != 0:
                    print("Error! Edge length must be integer value.")
                    sys.exit()

    def create_matrix(self):
        if self.config["grid_network"]:
            size = self.config["grid_size"]
            self.edge_list = []
            for x0 in range(size):
                for y0 in range(size):
                    for x1 in range(size):
                        for y1 in range(size):
                            self.edge_list.append(abs(x1 - x0) + abs(y1 - y0))
        else:
            self.edge_list = self.config["edge_lengths"]
        tmp = 0
        for i in range(self.node):
            for j in range(self.node):
                self.edge_matrix[i][j] = self.edge_list[tmp]
                self.bounds[j] = max(self.bounds[j], self.edge_matrix[i][j])
                self.price_factor[i][j] = self.edge_list[tmp]
                self.demand_factor[i][j] = -self.edge_list[tmp] ** 2
                tmp += 1

        if self.generate_skew:
            if self.config["grid_network"]:
                size = self.config["grid_size"]
                for x0 in range(size):
                    for y0 in range(size):
                        for x1 in range(size):
                            for y1 in range(size):
                                i = x0 + y0
                                j = x1 + y1
                                factor = 1
                                if i > j:
                                    factor = self.custom_skew
                                elif i < j:
                                    factor = 1 / self.custom_skew
                                self.demand_factor[i][j] *= factor
            else:
                for i in range(self.node):
                    for j in range(self.node):
                        if i > j:
                            factor = self.custom_skew
                        else:
                            factor = 1 / self.custom_skew
                        self.demand_factor[i][j] *= factor

    def step(self, act):
        if self.multi_agent:
            self.node_step(act)
            if self.current_index == self.node:
                reward, info = self.cycle_proceed()
                return self.to_observation(), reward, False, info
            return self.to_observation(), 0, False, {}
        else:
            for i in range(self.node):
                acti = act[i * self.node * 2:(i + 1) * self.node * 2]
                self.action_cache[i] = VehicleAction(self, i, acti)
            reward, info = self.cycle_proceed()
            return self.to_observation(), reward, False, info

    def cycle_step(self, act):
        for step in range(self.node):
            self.node_step(act[step])
        reward, info = self.cycle_proceed()
        return self.to_observation(), reward, False, info

    def node_step(self, act):
        action = VehicleAction(self, self.current_index, act)
        self.action_cache[self.current_index] = action
        self.current_index += 1

    def cycle_proceed(self):
        self.current_index = 0
        op_cost = 0
        wait_pen = 0
        overf = 0
        rew = 0
        reb = 0
        serve = 0

        stats_queue = 0
        stats_price = 0

        difference = 0
        if self.imitate:
            difference = self.calculate_imitation_reward()

        # Move cars in mini-nodes ahead
        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                # Sweeping BACKWARDS to avoid pushing vehicles multiple times in same time step
                for m in range(self.edge_matrix[i][j] - 1):
                    if m == 0:
                        # Stop tracking mini-node behavior and push cars to main node
                        self.vehicles[j] += self.mini_vehicles[i][j][m]
                    else:
                        # Vehicles still in mini nodes (traveling)
                        # Shifting vehicles further along path
                        self.mini_vehicles[i][j][m - 1] = self.mini_vehicles[i][j][m]
                    self.mini_vehicles[i][j][m] = 0

        for i in range(self.node):
            for j in range(self.node):
                if i == j:
                    continue
                veh_motion = self.action_cache[i].motion[j]
                # Statement to feed to mini-nodes
                # Only feed to mini nodes if required (edge length > 1)   ->   Feed to first mini-node
                if self.edge_matrix[i][j] > 1:
                    # for distance 2, it feeds to the 1st mininode (index 0)
                    # for distance 5, it feeds to the 4th mininode (index 3)
                    self.mini_vehicles[i][j][self.edge_matrix[i][j] - 2] += veh_motion
                else:
                    # Cars arriving at node j (for length 1 case)
                    self.vehicles[j] += veh_motion
                self.vehicles[i] -= veh_motion
                moved_customer = min(self.queue[i][j], veh_motion)
                self.queue[i][j] = self.queue[i][j] - moved_customer
                edge_len = self.edge_matrix[i][j]

                # statistics
                serve += edge_len * moved_customer
                op_cost += veh_motion * self.operating_cost * edge_len
                reb += (veh_motion - moved_customer) * self.operating_cost * edge_len

                price = self.action_cache[i].price[j]
                price_fac = self.price_factor[i][j]
                demand_fac = self.demand_factor[i][j]
                wait_pen += self.queue[i][j] * self.waiting_penalty
                freq = self.poisson_param * (price - 1) * price_fac / demand_fac
                request = min(self.poisson_cap, self.random.poisson(freq))
                act_req = request
                if self.queue[i][j] + act_req > self.queue_size:
                    act_req = 0
                overf += (request - act_req) * self.overflow
                self.queue[i][j] += act_req
                rew += act_req * price * price_fac
                stats_price += price
                stats_queue += self.queue[i][j]

        reward = rew - op_cost - wait_pen - overf
        current_reward = reward / self.reward_factor
        if self.use_average_reward:
            current_reward -= self.average_reward.average() / self.reward_factor
        self.average_reward.add(reward)
        if self.imitate:
            current_reward = difference
        debug_info = {'gain': rew, 'operating_cost': op_cost, 'wait_penalty': wait_pen, 'overflow': overf,
                      'reward': reward, 'price': stats_price / self.node / (self.node - 1),
                      'queue': stats_queue / self.node / (self.node - 1), 'imitation_reward': difference,
                      'rebalancing_cost': reb, 'distance_served': serve}
        return current_reward, debug_info

    def calculate_imitation_reward(self):
        diff = 0
        reference = self.imitation.compute_action()
        factor = self.vehicle
        for i in range(self.node):
            ref_action = VehicleAction(self, i, reference[i])
            self_action = self.action_cache[i]
            for j in range(self.node):
                diff += ((ref_action.motion[j] - self_action.motion[j]) / factor) ** 2
                diff += (ref_action.price[j] - self_action.price[j]) ** 2
        return -diff

    def reset(self):
        if self.average_queue_reset:
            self.average_reward.reset()
        # Reset queue, vehicles at nodes AND in travel
        for i in range(self.node):
            self.vehicles[i] = 0
            for j in range(self.node):
                self.queue[i][j] = 0
                for k in range(self.edge_matrix[i][j] - 1):
                    self.mini_vehicles[i][j][k] = 0

        for i in range(self.vehicle):
            pos = self.random.randint(0, self.node)
            self.vehicles[pos] = self.vehicles[pos] + 1
        self.over = 0
        self.current_index = 0
        return self.to_observation()

    def copy_from(self, env):
        self.vehicles = [env.vehicles[i] for i in range(self.node)]
        self.queue = [[env.queue[i][j] for j in range(self.node)] for i in range(self.node)]
        self.mini_vehicles = [[[env.mini_vehicles[i][j][k] for k in range(self.edge_matrix[i][j] - 1)]
                               for j in range(self.node)] for i in range(self.node)]

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def to_observation(self):
        if self.multi_agent:
            arr = [0 for _ in range(self.node * (3 + (1 if self.include_edge else 0) + self.mini_node_layer) + 1)]
            ind = 0
            for i in range(self.node):
                arr[ind] = self.vehicles[i]
                ind += 1
            for j in range(self.node):
                sums = [0 for _ in range(self.bounds[j] - 1)]
                for i in range(self.node):
                    for k in range(self.edge_matrix[i][j] - 1):
                        sums[k] += self.mini_vehicles[i][j][k]
                for i in range(self.mini_node_layer):
                    arr[ind] = self.mini_node_processor(sums, i, j)
                    ind += 1
            for i in range(self.node):
                arr[ind] = self.queue[self.current_index][i]
                ind += 1
            for i in range(self.node):
                arr[ind] = sum(self.queue[i])
                ind += 1
            if self.include_edge:
                for i in range(self.node):
                    arr[ind] = self.edge_matrix[self.current_index][i]
                    ind += 1
            arr[ind] = self.current_index
            return arr
        else:
            arr = [0 for _ in range(self.node * (self.node + 1 + self.mini_node_layer))]
            ind = 0
            for i in range(self.node):
                arr[ind] = self.vehicles[i]
                ind += 1
            for j in range(self.node):
                sums = [0 for _ in range(self.bounds[j] - 1)]
                for i in range(self.node):
                    for k in range(self.edge_matrix[i][j] - 1):
                        sums[k] += self.mini_vehicles[i][j][k]
                for i in range(self.mini_node_layer):
                    arr[ind] = self.mini_node_processor(sums, i, j)
                    ind += 1
            for i in range(self.node):
                for j in range(self.node):
                    arr[ind] = self.queue[i][j]
                    ind += 1
            return arr
