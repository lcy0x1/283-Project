import numpy as np


class Imitated:

    def __init__(self, env, dummy):
        self.env = env
        self.dummy = dummy

        self.time_factor = 0.5
        self.dist_factor = 0.3
        self.queue_factor = 0.4
        self.queue_intention = 1
        self.distribute_factor = 0.4
        self.price_intention = 0.25

        self.operating_cost = 0.6
        self.data_factor = 1

    def compute_action(self):
        vehicle_gradient = self.compute_gradient(self.env.vehicles, self.env.mini_vehicles, self.env.queue)
        # calculate action
        raw_motion = np.zeros((self.env.node, self.env.node))
        vehicle_motion = np.zeros((self.env.node, self.env.node))
        for i in range(self.env.node):
            if self.env.vehicles[i] == 0:
                continue
            intentions = sum(vehicle_gradient[i] * self.data_factor)
            factor = 1
            remain = self.env.vehicles[i] - intentions
            if intentions > self.env.vehicles[i]:
                factor = self.env.vehicles[i] / intentions
                remain = 0

            for j in range(self.env.node):
                raw_motion[i][j] = vehicle_gradient[i][j] * self.data_factor * factor

            raw_motion[i][i] = remain

            sums = sum(raw_motion[i])

            for j in range(self.env.node):
                vehicle_motion[i][j] = raw_motion[i][j] / sums

        # calculate price
        if self.env.skip_second_gradient:
            future_gradient = self.mimic_step(vehicle_gradient, raw_motion)
            departure = np.sum(raw_motion, 1)
            arrival = np.sum(raw_motion, 0)
            future_vehicle = np.array(self.env.vehicles) - departure + arrival
            future_queue = np.maximum(0, np.array(self.env.queue) - raw_motion)
        else:
            # imitate
            self.dummy.copy_from(self.env)
            for i in range(self.env.node):
                action = np.concatenate((vehicle_motion[i], np.ones(self.env.node)))
                self.dummy.node_step(action)
            self.dummy.cycle_proceed()
            future_gradient = self.compute_gradient(self.dummy.vehicles, self.dummy.mini_vehicles, self.dummy.queue)
            future_vehicle = self.dummy.vehicles
            future_queue = self.dummy.queue

        price = np.zeros((self.env.node, self.env.node))
        action = []
        for i in range(self.env.node):
            intentions = np.sum(future_gradient[i]) * self.data_factor
            factor = 1
            remain = future_vehicle[i] - intentions
            no_remain = remain / (self.env.node - 1)
            for j in range(self.env.node):
                intention = future_gradient[i][j] * self.data_factor * factor + no_remain - future_queue[i][j]
                price[i][j] = self.compute_best_price(intention)
            action.append(np.concatenate((vehicle_motion[i], price[i])))

        return action

    def compute_gradient(self, vehicles, mini_vehicles, queue):

        vehicle_potential = [0 for _ in range(self.env.node)]
        # compute vehicle availability
        # currently available vehicles - current queue +
        # upcoming vehicle * time factor ^ distance +
        # vehicle availability * queue ratio * queue_factor * time factor ^ distance
        for i in range(self.env.node):
            potential = 0
            potential -= sum(queue[i]) / self.data_factor
            potential += vehicles[i] / self.data_factor
            for j in range(self.env.node):
                for k in range(self.env.edge_matrix[j][i] - 1):
                    potential += mini_vehicles[j][i][k] * self.time_factor ** k / self.data_factor
            vehicle_potential[i] = potential

        # add queue potential
        for i in range(self.env.node):
            potential = 0
            for j in range(self.env.node):
                availability = queue[j][i] / self.data_factor
                time_discount = self.time_factor ** self.env.edge_matrix[j][i]
                potential += availability * self.queue_factor * time_discount
            vehicle_potential[i] += potential

        # calculate relative potential
        average_potential = sum(vehicle_potential) / self.env.node
        vehicle_diff = [0 for _ in range(self.env.node)]
        for i in range(self.env.node):
            vehicle_diff[i] = vehicle_potential[i] - average_potential

        # calculate vehicle gradient
        vehicle_gradient = [[0 for _ in range(self.env.node)] for _ in range(self.env.node)]
        for i in range(self.env.node):
            for j in range(self.env.node):
                diff = (vehicle_potential[i] - vehicle_potential[j]) * self.distribute_factor
                if diff > 0:
                    grad = diff * self.dist_factor ** self.env.edge_matrix[i][j] / self.env.node
                    vehicle_gradient[i][j] += grad
                vehicle_gradient[i][j] += queue[i][j] * self.queue_intention
        return vehicle_gradient

    def mimic_step(self, gradient, motion):
        ans = [[0 for _ in range(self.env.node)] for _ in range(self.env.node)]
        for i in range(self.env.node):
            for j in range(self.env.node):
                ans[i][j] = max(0, gradient[i][j] - motion[i][j])
        return ans

    def compute_best_price(self, intention):
        if intention < 0:
            return 1
        return max(self.operating_cost, 1 - intention * self.price_intention / self.data_factor)
