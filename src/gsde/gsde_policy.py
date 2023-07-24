from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from src.gsde.basemodel import SB3BaseModel as BaseModel
from src.gsde.distribution_helper import get_entropy, sum_independent_dims, get_mean
from torch.distributions import Normal
from torch import nn
from typing import Optional, Tuple


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
        self.create_feature_extractors()

        self.log_std_init = log_std_init
        self.should_learn_features = should_learn_features
        self.dist_kwargs = {
            "should_learn_features": should_learn_features,
        }

        self.action_size = int(
            np.prod(action_space.shape)
        )  # Number of action dimensions
        self.latent_sde_dim = None  # Dimension of the latent state dependent noise
        self.mean_actions = None  # Mean actions (initialized later)
        self.log_std = None  # Log standard deviation (initialized later)
        self.weights_dist = None  # Distribution of weights used for exploration
        self.exploration_mat = None  # Exploration matrix for single sample
        self.exploration_matrices = None  # Exploration matrices for multiple samples
        self._latent_sde = None  # Latent state-dependent encoding
        self.epsilon = 1e-6  # Small epsilon value for numerical stability
        self.should_learn_features = (
            should_learn_features  # Flag indicating whether to learn features
        )

        self._build(lr_schedule)

    def create_feature_extractors(self):
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

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
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        # Actions could be on arbitrary scale, so clip the actions to avoid
        # out of bound error (e.g. if sampling from a Gaussian distribution)
        actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def reset_noise(self, n_envs: int = 1, log_std=None) -> None:
        """
        Samples weights for exploration based on the provided log standard deviation.

        Args:
            log_std (th.Tensor): Logarithm of the standard deviation.
            batch_size (int): Number of weight samples to generate. Defaults to 1.

        Returns:
            None
        """
        if log_std is None:
            log_std = self.log_std
        std = self.get_std(log_std)

        # Create a distribution with mean zero and the computed standard deviation
        self.weights_dist = Normal(th.zeros_like(std), std)

        # Sample a single exploration matrix
        self.exploration_mat = self.weights_dist.rsample()
        # Sample multiple exploration matrices for batched exploration
        self.exploration_matrices = self.weights_dist.rsample((n_envs,))

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """For pickling purposes"""
        del progress_remaining
        return 0.0

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization
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
        self.action_net, self.log_std = self.proba_distribution_net(
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

    def compute_log_probs(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            hidden_pi, hidden_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            hidden_pi = self.mlp_extractor.forward_actor(pi_features)
            hidden_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(hidden_vf)
        distribution = self._get_action_dist_from_latent(hidden_pi)
        return values, distribution

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        values, distribution = self.compute_log_probs(obs, deterministic=deterministic)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        values, distribution = self.compute_log_probs(obs)
        log_prob = distribution.log_prob(actions)
        entropy = get_entropy(distribution)
        return values, log_prob, entropy

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
        return self.proba_distribution(mean_actions, self.log_std, latent_pi)

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def get_distribution(self, obs: th.Tensor):
        params = super().extract_features(obs, self.pi_features_extractor)
        hidden = self.mlp_extractor.forward_actor(params)
        return self._get_action_dist_from_latent(hidden)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        params = super().extract_features(obs, self.vf_features_extractor)
        hidden = self.mlp_extractor.forward_critic(params)
        return self.value_net(hidden)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return get_mean(self.distribution)
        return self.sample()

    def get_std(self, log_std: th.Tensor) -> th.Tensor:
        """
        Computes the standard deviation from the provided log standard deviation.

        Args:
            log_std (th.Tensor): Logarithm of the standard deviation.

        Returns:
            th.Tensor: Standard deviation tensor.
        """

        # Exponential scaling of standard deviation
        below_threshold = th.exp(log_std) * (log_std <= 0)
        safe_log_std = log_std * (log_std > 0) + self.epsilon
        above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
        std = below_threshold + above_threshold

        return std

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = -2.0,
        latent_sde_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Creates a probability distribution network for generating mean actions and log standard deviation.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            log_std_init (float): Initial value for the logarithm of the standard deviation.
                Defaults to -2.0.
            latent_sde_dim (Optional[int]): Dimensionality of the latent space for the state-dependent encoding.
                If None, the latent_dim value is used. Defaults to None.

        Returns:
            Tuple[nn.Module, nn.Parameter]: Tuple containing the mean actions network and the log standard deviation parameter.
        """
        mean_actions_net = nn.Linear(latent_dim, self.action_size)

        # Set the latent_sde_dim value to either the provided value or latent_dim
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim

        log_std = th.ones(self.latent_sde_dim, self.action_size)
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)

        # Sample weights for exploration based on the log standard deviation
        self.reset_noise(log_std=log_std)
        return mean_actions_net, log_std

    def actions_from_params(
        self,
        mean_actions: th.Tensor,
        log_std: th.Tensor,
        latent_sde: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        """
        Generates actions based on the mean actions, log standard deviation, and the latent state-dependent encoding.

        Args:
            mean_actions (th.Tensor): Mean actions tensor.
            log_std (th.Tensor): Logarithm of the standard deviation tensor.
            latent_sde (th.Tensor): Latent state-dependent encoding tensor.
            deterministic (bool): Flag indicating whether to generate deterministic actions.
                Defaults to False.

        Returns:
            th.Tensor: Generated actions tensor.
        """
        # Construct the probability distribution based on the provided parameters
        self.proba_distribution(mean_actions, log_std, latent_sde)
        # Generate actions from the distribution
        return self.get_actions(deterministic=deterministic)

    def proba_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ):
        """
        Constructs the probability distribution based on the mean actions, log standard deviation,
        and the latent state-dependent encoding.

        Args:
            mean_actions (th.Tensor): Mean actions tensor.
            log_std (th.Tensor): Logarithm of the standard deviation tensor.
            latent_sde (th.Tensor): Latent state-dependent encoding tensor.

        Returns:
            Distribution: Constructed probability distribution.
        """
        self._latent_sde = (
            latent_sde if self.should_learn_features else latent_sde.detach()
        )

        # Compute the variance based on the latent state-dependent encoding and the standard deviation
        variance = th.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        # Construct the distribution using mean actions and the square root of the variance
        self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Computes the log probability of actions in the probability distribution.

        Args:
            actions (th.Tensor): Actions tensor.

        Returns:
            th.Tensor: Log probability tensor.
        """
        gaussian_actions = actions

        # Compute the log probability of the Gaussian actions
        log_prob = self.distribution.log_prob(gaussian_actions)
        log_prob = sum_independent_dims(log_prob)

        return log_prob

    def sample(self) -> th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise

        return actions

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        """
        Computes the noise based on the latent state-dependent encoding.
        """
        # Detach the latent state-dependent encoding if features are not learned
        latent_sde = latent_sde if self.should_learn_features else latent_sde.detach()
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat.to(latent_sde.device))
        latent_sde = latent_sde.unsqueeze(dim=1)
        # Compute the noise based on the latent state-dependent encoding and the exploration matrices
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def log_prob_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Computes the log probability of actions based on the mean actions, log standard deviation,
        and the latent state-dependent encoding.

        Args:
            mean_actions (th.Tensor): Mean actions tensor.
            log_std (th.Tensor): Logarithm of the standard deviation tensor.
            latent_sde (th.Tensor): Latent state-dependent encoding tensor.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Tuple containing the generated actions tensor and the log probability tensor.
        """
        # Generate actions from the provided parameters
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        # Compute the log probability of the generated actions
        log_prob = self.log_prob(actions)
        return actions, log_prob
