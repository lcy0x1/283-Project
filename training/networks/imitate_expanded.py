from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from gym_route.envs.env import VehicleEnv


class InitParam(object):

    def __init__(self, env: VehicleEnv):
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
        veh = x[0:self.n]
        mini = x[self.n:self.n * (1 + self.mini)]
        queue = x[self.n * (1 + self.mini):self.n * (1 + self.mini + self.n)]
        return veh, mini, queue

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
            param.init(self.p0[i].weight.data[0], 1 / param.data_factor)
            for j in range(self.mini):
                param.init(self.p0[i].weight.data[1 + j], param.time_factor / param.data_factor)
            for j in range(self.n):
                param.init(self.p0[i].weight.data[1 + self.mini + j], -1 / param.data_factor)
            for j in range(self.n):
                factor = param.queue_factor * param.time_factor ** param.env.edge_matrix[j][i] / param.data_factor
                param.init(self.p0[i].weight.data[1 + self.mini + self.n + j], factor)
            param.init(self.p0[i].bias.data, 0)

        for i in range(self.n):
            for j in range(self.n):
                value = -1 / self.n
                if i == j:
                    value += 1
                param.init(self.p0[i].weight.data[i, j], value)
                param.init(self.p0[i].bias, 0)

    def get_potential(self, x: th.Tensor):
        l0 = []
        for i in range(self.n):
            res = th.concat((self.divider.vehicles(i, x),
                             self.divider.requests(i, x),
                             self.divider.arrivals(i, x)))
            l0.append(self.p0[i].forward(res))
        x0 = th.concat(l0)
        return self.p1.forward(x0)


class ActionNetwork(nn.Module):
    """
    Action:
    Linear mapping from queue + ReLU mapping from potential difference
    """

    def __init__(self, node: int, mini: int):
        super(ActionNetwork, self).__init__()

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

    def get_action(self, x: th.Tensor) -> List[th.Tensor]:
        potential = self.potential.get_potential(x)
        gradient = []
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
                sub[j] = val.view(1)
            sub[i] = self.relu.forward(remain)
            vec = th.concat(sub)
            gradient.append(vec / th.sum(vec))
        return gradient

    def get_price(self, x: th.Tensor) -> List[th.Tensor]:
        vehicle, mini, queue = self.divider.group(x)
        action = self.get_action(x)
        vehicle


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
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(ImitateNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
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


class CustomActorCriticPolicy(ActorCriticPolicy):
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
        super(CustomActorCriticPolicy, self).__init__(
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
