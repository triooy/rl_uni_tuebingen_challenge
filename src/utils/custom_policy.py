from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy as acp

import torch


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class ResidualNetwork(nn.Module):
    def __init__(
        self,
        res_blocks: int = 3,
        block_dims: List[int] = [64, 64, 64],
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.res_blocks = res_blocks
        self.block_dims = block_dims

        # Policy network
        self.policy_net = self.build_res_net(last_layer_dim_pi)
        # Value network
        self.value_net = self.build_res_net(last_layer_dim_vf)

    def build_res_block(
        self,
    ) -> nn.Module:
        # use self.block_dims to create a residual block
        input_dim = self.block_dims[0]
        net = []
        for i in range(1, len(self.block_dims)):
            output_dim = self.block_dims[i]
            net.append(nn.Linear(input_dim, output_dim))
            net.append(nn.ReLU())
            input_dim = output_dim
        return ResNet(nn.Sequential(*net))

    def build_res_net(self, last_layer) -> nn.Module:
        net = []
        for i in range(self.res_blocks):
            net.append(self.build_res_block())
        net.append(nn.Linear(self.block_dims[-1], last_layer))
        net.append(nn.ReLU())
        print(net)
        return nn.Sequential(*net)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class ResiudalActorCriticPolicy(acp):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ResidualNetwork(self.features_dim)
