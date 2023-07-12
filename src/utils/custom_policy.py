from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as t
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy as acp

import torch
import logging

logger = logging.getLogger(__name__)


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super(ResNet, self).__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class ResidualNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        res_blocks: Union[int, list] = 3,
        block_dims: List[int] = [64, 64, 64],
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.res_blocks = res_blocks
        self.block_dims = block_dims
        self.input_dim = feature_dim

        # Policy network
        self.policy_net = self.build_res_net(last_layer_dim_pi)
        # Value network
        self.value_net = self.build_res_net(last_layer_dim_vf)
        logger.info("ResidualNetwork initialized")

    def build_res_block(
        self,
        input_dim: int,
    ) -> nn.Module:
        # use self.block_dims to create a residual block
        net = []
        for i in range(0, len(self.block_dims)):
            output_dim = self.block_dims[i]
            net.append(nn.Linear(input_dim, output_dim))
            net.append(nn.ReLU())
            input_dim = output_dim
        return ResNet(nn.Sequential(*net))

    def build_res_net(self, last_layer) -> nn.Module:
        if isinstance(self.res_blocks, int):
            return self.build__resnet_1_connection(last_layer)
        elif isinstance(self.res_blocks, list):
            return self.build_resnet_2_connections(last_layer)
        else:
            raise ValueError("res_blocks must be either int or list")

    def forward(self, features: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: t.Tensor) -> t.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: t.Tensor) -> t.Tensor:
        return self.value_net(features)

    def build__resnet_1_connection(self, last_layer) -> nn.Module:
        net = [nn.Linear(self.input_dim, self.block_dims[0]), nn.ReLU()]
        input_dim = self.block_dims[0]
        for i in range(self.res_blocks):
            res_block = self.build_res_block(input_dim)
            net.append(res_block)
            input_dim = res_block.module[-2].out_features
        net.append(nn.Linear(input_dim, last_layer))
        net.append(nn.ReLU())
        return nn.Sequential(*net)

    def build_resnet_2_connections(self, last_layer) -> nn.Module:
        small_blocks = self.res_blocks[0]
        big_blocks = self.res_blocks[1]
        net = [nn.Linear(self.input_dim, self.block_dims[0]), nn.ReLU()]
        input_dim = self.block_dims[0]

        for i in range(small_blocks):
            inner_net = []
            for y in range(big_blocks):
                res_block = self.build_res_block(input_dim)
                inner_net.append(res_block)
                input_dim = res_block.module[-2].out_features
            inner_res = ResNet(nn.Sequential(*inner_net))
            net.append(inner_res)
            input_dim = inner_res.module[-1].module[-2].out_features
        net.append(nn.Linear(input_dim, last_layer))
        return nn.Sequential(*net)


class ResidualPolicy(acp):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self.res_blocks = kwargs.pop("res_blocks")
        self.block_dims = kwargs.pop("block_dims")
        self.last_layer_dim_pi = kwargs.pop("last_layer_dim_pi")
        self.last_layer_dim_vf = kwargs.pop("last_layer_dim_vf")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ResidualNetwork(
            self.features_dim,
            self.res_blocks,
            self.block_dims,
            self.last_layer_dim_pi,
            self.last_layer_dim_pi,
        )
