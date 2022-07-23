from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

from gym_route.envs.env import VehicleEnv

"""
This file contains everything for a imitate network setup
Debugger: helps to verify that the network works as intended
InitParam: initialize the parameters in certain way to verify expected behavior
FeatureDivider: a utility class to split the 1-dimensional input to several parts
PotentialNetwork: the network for vehicle potential. First part of 2 parts for action net
ActionNetwork: the network for vehicle movements and price. Second part of 2 parts for action net
ValueNetwork: value net implementation. Could be switched out for other implementation
ImitationNetwork: container class for ActionNetwork and ValueNetwork
ForwardNet: prevent output of ActionNetwork from going through another linear layer. A hack to StableBaselines
ImitateACP: a bridge between my network and StableBaselines ActorCriticPolicy
"""

""" holder for parameters """
dummy_env = VehicleEnv()


class Debugger(object):
    """
    helps to verify that the network works as intended
    """

    def __init__(self):
        self.envs: List[VehicleEnv] = []
        self.ep = 0

    def setup(self, x: th.Tensor):
        """
        copy value from tensor to environment
        """
        ep = x.shape[0]
        self.ep = ep
        self.envs = [VehicleEnv() for _ in range(ep)]
        for i in range(ep):
            self.setup_index(x[i], i)

    def setup_index(self, x: th.Tensor, index: int):
        env = self.envs[index]
        for i in range(env.node):
            env.vehicles[i] = x[i].item()
        for i in range(env.node):
            for j in range(env.node):
                env.queue[i][j] = x[env.node * (1 + env.mini_node_layer + i) + j].item()


class InitParam(object):
    """
    initialize the parameters in certain way to verify expected behavior
    """

    def __init__(self, env: VehicleEnv = dummy_env):
        self.time_factor = 0.5
        self.data_factor = 1
        self.queue_factor = 0.4
        self.dist_factor = 0.3
        self.distribute_factor = 0.2
        self.queue_intention = 1
        self.price_factor = 0.25
        self.arrival_factor = 1
        self.func = nn.init.constant_
        self.env = env

    def init(self, x: th.Tensor, mean: float):
        self.func(x, mean)


class FeatureDivider(nn.Module):
    """
    a utility class to split the 1-dimensional input to several parts.
    input data is of size [ep, len], where ep is episode,
    and len is the total length of observation
    """

    def __init__(self):
        super(FeatureDivider, self).__init__()

        n = dummy_env.node
        mini = dummy_env.mini_node_layer
        self.n = n
        self.mini = mini

        self.value_param_count = n * (4 + mini)

    def all_queue(self, x: th.Tensor) -> th.Tensor:
        """
        extract all queue information in dimension of [ep, n, n],
        where n is number of nodes
        """
        ep = x.shape[0]
        n = self.n
        m = self.mini
        return x[:, n * (1 + m):n * (1 + m + n)].view((ep, n, n))

    def all_vehicle(self, x: th.Tensor) -> th.Tensor:
        """
        extract all vehicle information in dimension of [ep, n]
        """
        return x[:, 0:self.n]

    def group(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        extract all vehicle information, mini-node vehicle information, and all queue information
        in dimension of [ep, n], [ep, n, m], and [ep, n, n]
        where m is the maximum visible depth of mini nodes
        """
        ep = x.shape[0]
        n = self.n
        m = self.mini
        mini = x[:, n:n * (1 + m)].view((ep, n, m))
        return self.all_vehicle(x), mini, self.all_queue(x)

    def potential_params(self, index: int, x: th.Tensor) -> th.Tensor:
        """
        extract information important for vehicle potential.
        This includes current available vehicle, future incoming vehicle,
        queue from here, and queue to here
        the output size is [ep, 1+m+n+n]
        """
        veh = x[:, index:index + 1]
        start = self.n + index * self.n
        mini = x[:, start:start + self.mini]
        queue = self.all_queue(x)
        req = queue[:, index, :]
        arr = queue[:, :, index]
        return th.concat((veh, mini, req, arr), 1)

    def value_params(self, index: int, x: th.Tensor) -> th.Tensor:
        """
        extract information important for value sub-network.
        This includes vehicle distribution, mini-node vehicle information,
        queue from here, and queue to here, and sum of queues for each node
        the output size is [ep, n+m*n+n+n+n] = [ep, n*(m+4)]
        """
        n = self.n
        m = self.mini
        veh = x[:, 0:n]
        mini = x[:, n:n * (1 + m)]
        queue = self.all_queue(x)
        sums = th.sum(queue, 2)
        req = queue[:, index, :]
        arr = queue[:, :, index]
        return th.concat((veh, mini, req, arr, sums), 1)


class PotentialNetwork(nn.Module):
    """
    This is a linear layer, but with reduced parameters
    There are 18*8+8*8=208 parameters
    Could be replaced with a linear layer, which has 640 parameters
    This implementation has 1/3 parameters for 8-node network, and
    for n-node network, it has 3*n^2 parameters, while linear layer has 2*n^3 parameters

    Logic:
    For each node, potential parameters (size 18) forward to 1 value (potential)
    Then, all values go through a linear layer (8x8) to support more complicated logic
    """

    def __init__(self):
        super(PotentialNetwork, self).__init__()

        node = dummy_env.node
        mini = dummy_env.mini_node_layer
        self.n = node
        self.mini = mini
        self.divider = FeatureDivider()

        # n parallel layers of size x*1
        self.p0 = [nn.Linear(1 + mini + node * 2, 1) for _ in range(node)]
        # 1 layer of size n*n
        self.p1 = nn.Linear(node, node)

        # register layers so they are initialized
        for i in range(node):
            self.add_module('p0_' + str(i), self.p0[i])
        self.add_module('p1', self.p1)

    def init(self, param: InitParam):
        """
        initialize network value to mimic algorithm. Not mandatory
        """
        for i in range(self.n):
            param.init(self.p0[i].weight.data[0][0], 1 / param.data_factor)
            for j in range(self.mini):
                param.init(self.p0[i].weight.data[0][1 + j], param.time_factor / param.data_factor)
            for j in range(self.n):
                param.init(self.p0[i].weight.data[0][1 + self.mini + j], -1 / param.data_factor)
            for j in range(self.n):
                factor = param.queue_factor * param.time_factor ** param.env.edge_matrix[j][i] / param.data_factor
                param.init(self.p0[i].weight.data[0][1 + self.mini + self.n + j], factor)
            param.init(self.p0[i].bias.data, 0)

        for i in range(self.n):
            for j in range(self.n):
                value = -1 / self.n
                if i == j:
                    value += 1
                param.init(self.p1.weight.data[i, j], value)
                param.init(self.p1.bias, 0)

    def get_potential(self, x: th.Tensor):
        """
        forward function
        """
        l0 = []
        # forward for parallel layers
        for i in range(self.n):
            l0.append(self.p0[i].forward(self.divider.potential_params(i, x)))
        # concat the independent values and then forward for the second layer
        x0 = th.concat(l0, 1)
        return self.p1.forward(x0)

    def apply(self, fn):
        """
        when initializing network, set parameters as well
        """
        super(PotentialNetwork, self).apply(fn)
        self.init(InitParam())


class ActionNetwork(nn.Module):
    """
    Action:
    Linear mapping from queue + ReLU mapping from potential difference
    """

    def __init__(self, debug: bool = False):
        super(ActionNetwork, self).__init__()
        node = dummy_env.node
        mini = dummy_env.mini_node_layer
        self.debug = debug
        self.n = node
        self.mini = mini
        self.potential = PotentialNetwork()
        self.divider = FeatureDivider()
        self.relu = nn.ReLU()

        # initialize parameters
        self.distribute_param = nn.Parameter(th.empty((node, node)))
        self.queue_param = nn.Parameter(th.empty((node, node)))
        self.departure_factor = nn.Parameter(th.empty((node, node)))
        self.arrival_factor = nn.Parameter(th.empty((node, node)))
        self.mini_factor = nn.Parameter(th.empty((node, mini)))
        self.intention_factor = nn.Parameter(th.empty((node, node)))
        self.price_factor = nn.Parameter(th.empty((node, node)))

        # register sub-modules and parameters
        self.add_module("potential", self.potential)
        self.register_parameter('distribute_param', self.distribute_param)
        self.register_parameter('queue_param', self.queue_param)
        self.register_parameter('departure_factor', self.departure_factor)
        self.register_parameter('arrival_factor', self.arrival_factor)
        self.register_parameter('mini_factor', self.mini_factor)
        self.register_parameter('intention_factor', self.intention_factor)
        self.register_parameter('price_factor', self.price_factor)

    def init(self, param: InitParam):
        """
        initialize network value to mimic algorithm. Not mandatory
        """
        for i in range(self.n):
            for j in range(self.n):
                dist_fac = param.dist_factor ** param.env.edge_matrix[j][i]
                value = dist_fac * param.distribute_factor / self.n / param.data_factor
                param.init(self.distribute_param.data[i, j], value)

        for i in range(self.n):
            for j in range(self.n):
                value = param.queue_intention / param.data_factor
                param.init(self.queue_param.data[i, j], value)

        for i in range(self.n):
            for j in range(self.n):
                param.init(self.departure_factor.data[i, j], 1)

        for i in range(self.n):
            for j in range(self.n):
                dist_fac = param.arrival_factor ** param.env.edge_matrix[j][i]
                param.init(self.arrival_factor.data[i, j], dist_fac)

        for i in range(self.n):
            for j in range(self.mini):
                param.init(self.mini_factor.data[i, j], 1)

        for i in range(self.n):
            for j in range(self.n):
                param.init(self.intention_factor.data[i, j], param.data_factor)

        for i in range(self.n):
            for j in range(self.n):
                param.init(self.price_factor.data[i, j], param.price_factor)

    def get_action(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # get vehicle potential
        potential = self.potential.get_potential(x)
        # get vehicle potential gradient. size [ep, n, n], diagonal is always 0
        diff = potential.unsqueeze(2) - potential.unsqueeze(1)
        # get ReLU of gradient to destroy symmetry and remove negative values
        val_0 = self.relu.forward(diff) * self.distribute_param
        # add queue to positive gradient. Use parameter to achieve weighted sum
        val_1 = self.divider.all_queue(x) * self.queue_param
        # this is desired vehicle movement when having abundant vehicle
        grad = val_0 + val_1
        # calculate remaining idle vehicle
        remain = self.divider.all_vehicle(x) - th.sum(grad, 2)
        # negative remaining vehicle is not allowed
        remain = self.relu.forward(remain)
        # set idle vehicle to diagonal, replacing zeros
        vec = grad[:, :, :]
        for i in range(x.shape[0]):
            vec[i] += th.diag(remain[i])
        # get total wanted vehicle, minimum is available vehicle. add 1e-3 to prevent div by zero
        tot = self.relu(th.sum(vec, 2) - 1e-3) + 1e-3
        # divide desired vehicle movement (include idle on diagonal) by total wanted.
        # Useful only when the available vehicle is not enough
        action = vec / tot.unsqueeze(2)
        # divide raw vehicle movement by total wanted.
        # Useful only when the available vehicle is not enough
        raw_action = grad / tot.unsqueeze(2)
        return grad, action, raw_action

    def get_price(self, x: th.Tensor):
        # get current state information
        vehicle, mini, queue = self.divider.group(x)
        # get calculated vehicle movement
        gradient, action, raw_action = self.get_action(x)
        # convert vehicle movement to absolute value
        raw_action = raw_action * vehicle.unsqueeze(2)
        # estimate next step vehicle gradient
        future_gradient = self.relu(gradient - raw_action)
        # estimation values for departure and arrivals of vehicle
        departure = th.sum(raw_action * self.departure_factor, 2)
        arrival = th.sum(raw_action * self.arrival_factor, 1) + th.sum(mini * self.mini_factor, 2)
        # estimate next step vehicle distribution
        future_vehicle = vehicle - departure + arrival
        # estimate next step queue assuming price of 1
        future_queue = th.relu(queue - raw_action)
        # calculate vehicle re-balancing intention (without queue)
        intentions = th.sum(future_gradient * self.intention_factor, 2)
        # calculate remaining idle vehicle
        remain = self.relu(future_vehicle - intentions)
        no_remain = remain / (self.n - 1)
        # calculate total vehicle intention (remove required vehicle for queue)
        intention = future_gradient + no_remain.unsqueeze(2) - future_queue
        # vary price based on intended empty vehicle movement and availability of idle vehicles
        price = self.relu(1 - self.relu(intention * self.price_factor) - 0.6) + 0.6
        return th.concat((action, price), 2).flatten(1)

    def forward(self, x: th.Tensor):
        ans = self.get_price(x)
        return ans

    def apply(self, fn):
        super(ActionNetwork, self).apply(fn)
        self.init(InitParam())


class ValueNetwork(nn.Module):
    """
    Value net. This use n parallel blocks of 2 layer
    """

    def __init__(self):
        super().__init__()
        middle = dummy_env.config["vfn_middle"]
        m = dummy_env.config["vfn_out"]
        self.n = dummy_env.node
        self.divider = FeatureDivider()
        self.output_size = self.n * m
        self.nets = [nn.Sequential(nn.Linear(self.divider.value_param_count, middle),
                                   nn.ReLU(), nn.Linear(middle, m), nn.ReLU())
                     for _ in range(self.n)]
        for i in range(self.n):
            self.add_module("net_" + str(i), self.nets[i])

    def forward(self, x: th.Tensor):
        results: List[Optional[th.Tensor]] = [None for _ in range(self.n)]
        for i in range(self.n):
            results[i] = self.nets[i].forward(self.divider.value_params(i, x))
        return th.concat(results, 1)


class ImitateNetwork(nn.Module):

    def __init__(self, feature_dim: int, env: VehicleEnv = dummy_env):
        super(ImitateNetwork, self).__init__()

        assert feature_dim == env.node * (env.node + env.mini_node_layer + 1), 'feature dimension mismatch'

        # Policy network
        self.policy_net = ActionNetwork()
        # Value network
        self.value_net = ValueNetwork()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = env.node ** 2 * 2
        self.latent_dim_vf = self.value_net.output_size

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ForwardNet(nn.Module):

    def __init__(self):
        super(ForwardNet, self).__init__()

    def forward(self, x: th.Tensor):
        # x = x.detach()
        return x


class ImitateACP(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(ImitateACP, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def re_init(self):
        param = InitParam()
        self.mlp_extractor.policy_net.init(param)
        self.mlp_extractor.policy_net.potential.init(param)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ImitateNetwork(self.features_dim)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        # skip last action layer
        self.action_net = ForwardNet()

        # skip training
        # nn.init.constant_(self.log_std.data, -10)
