from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from gym_route.envs.env import VehicleEnv

dummy_env = VehicleEnv()


class Debugger(object):

    def __init__(self):
        self.envs: List[VehicleEnv] = []
        self.ep = 0

    def setup(self, x: th.Tensor):
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

    def __init__(self, env: VehicleEnv = dummy_env):
        self.time_factor = 0.5
        self.data_factor = 1
        self.queue_factor = 0.4
        self.dist_factor = 0.3
        self.distribute_factor = 0.2
        self.queue_intention = 1
        self.func = nn.init.constant_
        self.env = env

    def init(self, x: th.Tensor, mean: float):
        self.func(x, mean)


class FeatureDivider(nn.Module):

    def __init__(self, n: int, mini: int):
        super(FeatureDivider, self).__init__()
        self.n = n
        self.mini = mini

    def group(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        ep = x.shape[1]
        veh = x[0:self.n]
        mini = x[self.n:self.n * (1 + self.mini)].view((self.n, self.mini, ep))
        queue = [x[self.n * (1 + self.mini + i):self.n * (1 + self.mini + i + 1)].unsqueeze(0) for i in range(self.n)]
        return veh, mini, th.concat(queue)

    def vehicles(self, index: int, x: th.Tensor) -> th.Tensor:
        veh = x[[index]]
        mini = x[[self.n + index * self.n + i for i in range(self.mini)]]
        return th.concat((veh, mini))

    def requests(self, index: int, x: th.Tensor) -> th.Tensor:
        return x[[self.n * (1 + self.mini) + index * self.n + i for i in range(self.n)]]

    def arrivals(self, index: int, x: th.Tensor) -> th.Tensor:
        return x[[self.n * (1 + self.mini) + i * self.n + index for i in range(self.n)]]

    def queue(self, x: th.Tensor, i: int, j: int):
        return x[self.n * (1 + self.mini) + i * self.n + j]


class PotentialNetwork(nn.Module):
    """
    This is a linear layer, but with reduced parameters
    There are 18*8+8*8=208 parameters
    Could be replaced with a linear layer, which has 640 parameters
    """

    def __init__(self, node: int, mini: int):
        super(PotentialNetwork, self).__init__()
        self.n = node
        self.mini = mini
        self.divider = FeatureDivider(node, mini)

        self.p0 = [nn.Linear(1 + mini + node * 2, 1) for _ in range(node)]
        self.p1 = nn.Linear(node, node)
        for i in range(node):
            self.add_module('p0_' + str(i), self.p0[i])
        self.add_module('p1', self.p1)

    def init(self, param: InitParam):
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
        l0 = []
        for i in range(self.n):
            res = th.concat((self.divider.vehicles(i, x),
                             self.divider.requests(i, x),
                             self.divider.arrivals(i, x)))
            l0.append(self.use_linear(self.p0[i], res))
        x0 = th.concat(l0)
        return self.use_linear(self.p1, x0)

    @staticmethod
    def use_linear(p, x):
        return th.swapaxes(p.forward(th.swapaxes(x, 0, 1)), 0, 1)

    def apply(self, fn):
        super(PotentialNetwork, self).apply(fn)
        self.init(InitParam())


class ActionNetwork(nn.Module):
    """
    Action:
    Linear mapping from queue + ReLU mapping from potential difference
    """

    def __init__(self, node: int, mini: int, debug: bool = False):
        super(ActionNetwork, self).__init__()

        self.debug = debug
        self.n = node
        self.potential = PotentialNetwork(node, mini)
        self.divider = FeatureDivider(node, mini)
        self.relu = nn.ReLU()

        self.distribute_param = nn.Parameter(th.empty((node, node)))
        self.queue_param = nn.Parameter(th.empty((node, node)))

        self.add_module("potential", self.potential)
        self.register_parameter('distribute_param', self.distribute_param)
        self.register_parameter('queue_param', self.queue_param)

    def init(self, param: InitParam):
        for i in range(self.n):
            for j in range(self.n):
                value = param.dist_factor ** param.env.edge_matrix[j][i] * param.distribute_factor / self.n
                param.init(self.distribute_param.data[i, j], value)

        for i in range(self.n):
            for j in range(self.n):
                value = param.queue_intention
                param.init(self.queue_param.data[i, j], value)

    def get_action(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        potential = self.potential.get_potential(x)
        gradient = []
        action = []
        raw_action = []
        for i in range(self.n):
            sub: List[Optional[th.Tensor]] = [None for _ in range(self.n)]
            remain = x[i]
            for j in range(self.n):
                if i == j:
                    continue
                diff = potential[i] - potential[j]
                val_0 = self.relu.forward(diff) * self.distribute_param[i, j]
                val_1 = self.divider.queue(x, i, j) * self.queue_param[i, j]
                val: th.Tensor = val_0 + val_1
                remain = remain - val
                sub[j] = val.unsqueeze(0)
            sub[i] = th.zeros(potential[0].shape).unsqueeze(0)
            grad = th.concat(sub)
            gradient.append(grad.unsqueeze(0))
            sub[i] = self.relu.forward(remain).unsqueeze(0)
            vec = th.concat(sub)
            action.append((vec / th.sum(vec, 0)).unsqueeze(0))
            raw_action.append((grad / th.sum(vec, 0)).unsqueeze(0))
        return th.concat(gradient), th.concat(action), th.concat(raw_action)

    # TODO add parameters
    def get_price(self, x: th.Tensor):
        vehicle, mini, queue = self.divider.group(x)
        gradient, action, raw_action = self.get_action(x)
        raw_action = th.swapaxes(th.swapaxes(raw_action, 0, 1) * vehicle, 0, 1)
        future_gradient = self.relu(gradient - raw_action)
        departure = th.sum(raw_action, 1)  # TODO parameter
        arrival = th.sum(raw_action, 0) + th.sum(mini, 1)  # TODO parameter
        future_vehicle = vehicle - departure + arrival
        future_queue = th.relu(queue - raw_action)

        ans = []
        intentions = th.sum(future_gradient, 1)  # TODO parameter
        remain = self.relu(future_vehicle - intentions)
        for i in range(self.n):
            no_remain = remain[i] / (self.n - 1)
            price = []
            for j in range(self.n):
                intention = future_gradient[i][j] + no_remain - future_queue[i][j]  # TODO parameter
                price.append((self.relu(1 - self.relu(intention) * 0.25 - 0.6) + 0.6).unsqueeze(0))  # TODO parameter
            ans.append(th.concat((action[i], th.concat(price))))
        return th.concat(ans)

    def forward(self, x: th.Tensor):
        ans = th.swapaxes(self.get_price(th.swapaxes(x, 0, 1)), 0, 1)
        if self.debug:
            debugger = Debugger()
            debugger.setup(x)
            for t in range(debugger.ep):
                act = debugger.envs[t].imitation.compute_action()
                trans = th.tensor(np.array(act)).flatten()
                diff = th.sum(th.abs(ans[t] - trans)).item()
                if diff > 10:
                    print(f'episode {t}: ')
                    print(diff)
                    print(ans[t].view((8, 16)))
                    print(trans.view((8, 16)))
        return ans

    def apply(self, fn):
        super(ActionNetwork, self).apply(fn)
        self.init(InitParam())


class ImitateNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 128,
            last_layer_dim_vf: int = 128,
    ):
        super(ImitateNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = ActionNetwork(8, 1)
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf), nn.ReLU()
        )

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

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ImitateNetwork(self.features_dim)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        nn.init.constant_(self.action_net.weight.data, 0)
        for i in range(self.mlp_extractor.latent_dim_pi):
            nn.init.constant_(self.action_net.weight.data[i, i], 1)
