import torch as t
from torch.distributions import Normal
from torch import nn
from typing import Optional, Tuple
from src.gsde.distribution_helper import sum_independent_dims, get_mean


class GeneralizedStateDependentDistribution:
    """
    GeneralizedStateDependentDistribution is a class representing a noise distribution
    that is dependent on the state. It extends the Distribution class.

        Args:
            action_size (int): The number of action dimensions.
            should_learn_features (bool): Flag indicating whether to learn features. Defaults to False.
            gsde_epsilon (float): A small epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(
        self,
        action_size: int,
        should_learn_features: bool = False,
        gsde_epsilon: float = 1e-6,
    ):
        self.action_size = action_size  # Number of action dimensions
        self.latent_sde_dim = None  # Dimension of the latent state dependent noise
        self.mean_actions = None  # Mean actions (initialized later)
        self.log_std = None  # Log standard deviation (initialized later)
        self.weights_dist = None  # Distribution of weights used for exploration
        self.exploration_mat = None  # Exploration matrix for single sample
        self.exploration_matrices = None  # Exploration matrices for multiple samples
        self._latent_sde = None  # Latent state-dependent encoding
        self.epsilon = gsde_epsilon  # Small epsilon value for numerical stability
        self.should_learn_features = (
            should_learn_features  # Flag indicating whether to learn features
        )

    def get_actions(self, deterministic: bool = False) -> t.Tensor:
        if deterministic:
            return get_mean(self.distribution)
        return self.sample()

    def get_std(self, log_std: t.Tensor) -> t.Tensor:
        """
        Computes the standard deviation from the provided log standard deviation.

        Args:
            log_std (t.Tensor): Logarithm of the standard deviation.

        Returns:
            t.Tensor: Standard deviation tensor.
        """

        # Exponential scaling of standard deviation
        below_threshold = t.exp(log_std) * (log_std <= 0)
        safe_log_std = log_std * (log_std > 0) + self.epsilon
        above_threshold = (t.log1p(safe_log_std) + 1.0) * (log_std > 0)
        std = below_threshold + above_threshold

        return std

    def sample_weights(self, log_std: t.Tensor, batch_size: int = 1) -> None:
        """
        Samples weights for exploration based on the provided log standard deviation.

        Args:
            log_std (t.Tensor): Logarithm of the standard deviation.
            batch_size (int): Number of weight samples to generate. Defaults to 1.

        Returns:
            None
        """
        std = self.get_std(log_std)

        # Create a distribution with mean zero and the computed standard deviation
        self.weights_dist = Normal(t.zeros_like(std), std)

        # Sample a single exploration matrix
        self.exploration_mat = self.weights_dist.rsample()
        # Sample multiple exploration matrices for batched exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

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

        log_std = t.ones(self.latent_sde_dim, self.action_size)
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)

        # Sample weights for exploration based on the log standard deviation
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def actions_from_params(
        self,
        mean_actions: t.Tensor,
        log_std: t.Tensor,
        latent_sde: t.Tensor,
        deterministic: bool = False,
    ) -> t.Tensor:
        """
        Generates actions based on the mean actions, log standard deviation, and the latent state-dependent encoding.

        Args:
            mean_actions (t.Tensor): Mean actions tensor.
            log_std (t.Tensor): Logarithm of the standard deviation tensor.
            latent_sde (t.Tensor): Latent state-dependent encoding tensor.
            deterministic (bool): Flag indicating whether to generate deterministic actions.
                Defaults to False.

        Returns:
            t.Tensor: Generated actions tensor.
        """
        # Construct the probability distribution based on the provided parameters
        self.proba_distribution(mean_actions, log_std, latent_sde)
        # Generate actions from the distribution
        return self.get_actions(deterministic=deterministic)

    def proba_distribution(
        self, mean_actions: t.Tensor, log_std: t.Tensor, latent_sde: t.Tensor
    ):
        """
        Constructs the probability distribution based on the mean actions, log standard deviation,
        and the latent state-dependent encoding.

        Args:
            mean_actions (t.Tensor): Mean actions tensor.
            log_std (t.Tensor): Logarithm of the standard deviation tensor.
            latent_sde (t.Tensor): Latent state-dependent encoding tensor.

        Returns:
            Distribution: Constructed probability distribution.
        """
        self._latent_sde = (
            latent_sde if self.should_learn_features else latent_sde.detach()
        )

        # Compute the variance based on the latent state-dependent encoding and the standard deviation
        variance = t.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        # Construct the distribution using mean actions and the square root of the variance
        self.distribution = Normal(mean_actions, t.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: t.Tensor) -> t.Tensor:
        """
        Computes the log probability of actions in the probability distribution.

        Args:
            actions (t.Tensor): Actions tensor.

        Returns:
            t.Tensor: Log probability tensor.
        """
        gaussian_actions = actions

        # Compute the log probability of the Gaussian actions
        log_prob = self.distribution.log_prob(gaussian_actions)
        log_prob = sum_independent_dims(log_prob)

        return log_prob

    def sample(self) -> t.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise

        return actions

    def get_noise(self, latent_sde: t.Tensor) -> t.Tensor:
        """
        Computes the noise based on the latent state-dependent encoding.
        """
        # Detach the latent state-dependent encoding if features are not learned
        latent_sde = latent_sde if self.should_learn_features else latent_sde.detach()
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return t.mm(latent_sde, self.exploration_mat.to(latent_sde.device))
        latent_sde = latent_sde.unsqueeze(dim=1)
        # Compute the noise based on the latent state-dependent encoding and the exploration matrices
        noise = t.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def log_prob_from_params(
        self, mean_actions: t.Tensor, log_std: t.Tensor, latent_sde: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        """
        Computes the log probability of actions based on the mean actions, log standard deviation,
        and the latent state-dependent encoding.

        Args:
            mean_actions (t.Tensor): Mean actions tensor.
            log_std (t.Tensor): Logarithm of the standard deviation tensor.
            latent_sde (t.Tensor): Latent state-dependent encoding tensor.

        Returns:
            Tuple[t.Tensor, t.Tensor]: Tuple containing the generated actions tensor and the log probability tensor.
        """
        # Generate actions from the provided parameters
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        # Compute the log probability of the generated actions
        log_prob = self.log_prob(actions)
        return actions, log_prob
