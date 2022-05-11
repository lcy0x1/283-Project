import torch as th
from torch import nn


class DataHolder(object):

    def __init__(self, x: th.Tensor):
        self.n = 8
        self.mini = 1

        x = th.swapaxes(x, 0, 1)
        self.vehicles = th.tensor([x[i] for i in range(self.n)])
        self.mini = th.tensor([x[i + self.n] for i in range(self.n)])
        self.queue = th.tensor([[x[i * self.n + j + self.n * 2] for j in range(self.n)] for i in range(self.n)])


class Imitated(nn.Module):

    def __init__(self):
        super(Imitated, self).__init__()
        self.time_factor = nn.Parameter(th.tensor(0.5))
        self.dist_factor = nn.Parameter(th.tensor(0.3))
        self.queue_factor = nn.Parameter(th.tensor(0.4))
        self.queue_intention = nn.Parameter(th.tensor(1))
        self.distribute_factor = nn.Parameter(th.tensor(0.4))
        self.price_intention = nn.Parameter(th.tensor(0.25))
        self.operating_cost = nn.Parameter(th.tensor(0.6))
        self.data_factor = nn.Parameter(th.tensor(1))

        self.register_parameter("time", self.time_factor)
        self.register_parameter("dist", self.dist_factor)
        self.register_parameter("queue", self.queue_factor)
        self.register_parameter("intention", self.queue_intention)
        self.register_parameter("distribute", self.distribute_factor)
        self.register_parameter("price", self.price_intention)
        self.register_parameter("cost", self.operating_cost)
        self.register_parameter("data", self.data_factor)

    def init(self):
        nn.init.constant_(self.time_factor.data, 0.5)
        nn.init.constant_(self.dist_factor.data, 0.3)
        nn.init.constant_(self.queue_factor.data, 0.4)
        nn.init.constant_(self.queue_intention.data, 1)
        nn.init.constant_(self.distribute_factor.data, 0.4)
        nn.init.constant_(self.price_intention.data, 0.25)
        nn.init.constant_(self.operating_cost.data, 0.6)
        nn.init.constant_(self.data_factor.data, 1)

    def compute_action(self, x: th.Tensor):
        data = DataHolder(x)
        vehicle_gradient = self.compute_gradient(data)
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

        future_gradient = self.mimic_step(vehicle_gradient, raw_motion)
        departure = np.sum(raw_motion, 1)
        arrival = np.sum(raw_motion, 0)
        future_vehicle = np.array(self.env.vehicles) - departure + arrival
        future_queue = np.maximum(0, np.array(self.env.queue) - raw_motion)

        price = np.zeros((self.env.node, self.env.node))
        action = []
        for i in range(self.env.node):
            intentions = np.sum(future_gradient[i]) * self.data_factor
            factor = 1
            remain = future_vehicle[i] - intentions
            if intentions > future_vehicle[i]:
                factor = future_vehicle[i] / intentions
                remain = 0
            no_remain = remain / (self.env.node - 1)
            for j in range(self.env.node):
                intention = future_gradient[i][j] * self.data_factor * factor + no_remain - future_queue[i][j]
                price[i][j] = self.compute_best_price(intention)
            action.append(np.concatenate((vehicle_motion[i], price[i])))

        return action

    def compute_gradient(self, data: DataHolder):
        vehicle_potential = [0 for _ in range(data.n)]

        for i in range(data.n):
            potential = 0
            potential -= th.sum(data.queue[i], 0) / self.data_factor
            potential += data.vehicles[i] / self.data_factor
            potential += data.mini[i] * self.time_factor / self.data_factor
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
                ans[i][j] = gradient[i][j] - motion[i][j]
        return ans

    def compute_best_price(self, intention):
        if intention < 0:
            return 1
        return max(self.operating_cost, 1 - intention * self.price_intention / self.data_factor)
