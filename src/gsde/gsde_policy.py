import collections
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from src.gsde.basemodel import SB3BaseModel as BaseModel
from src.gsde.gsde import GeneralizedStateDependentDistribution


class PPO_gSDE_MlpPolicy(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = dict(
            pi=[64, 64], vf=[64, 64]
        ),
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        should_learn_features: bool = False,
        share_features_extractor: bool = True,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        self.should_learn_features = should_learn_features
        self.dist_kwargs = {
            "should_learn_features": should_learn_features,
        }

        self.action_dist = GeneralizedStateDependentDistribution(
            int(np.prod(action_space.shape)),
            **self.dist_kwargs,
        )

        self._build(lr_schedule)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        epsiode_start=None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        self.set_training_mode(False)
        obs, env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        # Actions could be on arbitrary scale, so clip the actions to avoid
        # out of bound error (e.g. if sampling from a Gaussian distribution)
        actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def reset_noise(self, n_envs: int = 1) -> None:
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        parameters = super()._get_constructor_parameters()

        parameters.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=True,
                log_std_init=self.log_std_init,
                lr_schedule=self._dummy_schedule,
                ortho_init=self.ortho_init,
                optimizer_kwargs={"eps": 1e-5},
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return parameters

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi,
            latent_sde_dim=latent_dim_pi,
            log_std_init=self.log_std_init,
        )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **{"eps": 1e-5}
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def extract_features(
        self, obs: th.Tensor
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor)
        else:
            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(
            mean_actions, self.log_std, latent_pi
        )

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            hidden_pi, hidden_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            hidden_pi = self.mlp_extractor.forward_actor(pi_features)
            hidden_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(hidden_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(hidden_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        params = super().extract_features(obs, self.pi_features_extractor)
        hidden = self.mlp_extractor.forward_actor(params)
        return self._get_action_dist_from_latent(hidden)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        params = super().extract_features(obs, self.vf_features_extractor)
        hidden = self.mlp_extractor.forward_critic(params)
        return self.value_net(hidden)
