from stable_baselines3.common.distributions import Distribution
import torch as t
from torch.distributions import Normal
from torch import nn
from typing import Optional, Tuple


class Squasher:
    """
    The Squasher class is used to squash the output of the policy network using tanh.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: t.Tensor) -> t.Tensor:
        """
        Squashes the input tensor using the hyperbolic tangent (tanh) function.

        Args:
            x (t.Tensor): Input tensor.

        Returns:
            t.Tensor: Squashed tensor.
        """
        return t.tanh(x)

    @staticmethod
    def atanh(x: t.Tensor) -> t.Tensor:
        """
        Computes the inverse hyperbolic tangent (atanh) of the input tensor.

        Args:
            x (t.Tensor): Input tensor.

        Returns:
            t.Tensor: Inverse hyperbolic tangent tensor.
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: t.Tensor) -> t.Tensor:
        """
        Computes the inverse of the squashing operation by applying the inverse hyperbolic tangent (atanh)
        to the input tensor.

        Args:
            y (t.Tensor): Input tensor.

        Returns:
            t.Tensor: Inverse squashed tensor.
        """
        eps = t.finfo(y.dtype).eps
        return Squasher.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: t.Tensor) -> t.Tensor:
        """
        Computes the log probability correction term for the squashed tensor.

        Args:
            x (t.Tensor): Input tensor.

        Returns:
            t.Tensor: Log probability correction tensor.
        """
        # Squash correction
        return t.log(1.0 - t.tanh(x) ** 2 + self.epsilon)


class StateDependentNoiseDistribution(Distribution):
    """
    StateDependentNoiseDistribution is a class representing a noise distribution
    that is dependent on the state. It extends the Distribution class.

        Args:
            action_size (int): The number of action dimensions.
            use_full_std (bool): Flag indicating whether to use full standard deviation.
                Defaults to True.
            use_expln (bool): Flag indicating whether to use exponential scaling. Defaults to False.
            squash_output (bool): Flag indicating whether to squash the output using a squasher.
                Defaults to False.
            should_learn_features (bool): Flag indicating whether to learn features. Defaults to False.
            gsde_epsilon (float): A small epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(
        self,
        action_size: int,
        use_full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        should_learn_features: bool = False,
        gsde_epsilon: float = 1e-6,
    ):
        # Initialize the StateDependentNoiseDistribution with the provided parameters
        super().__init__()
        self.action_size = action_size  # Number of action dimensions
        self.latent_sde_dim = None  # Dimension of the latent state dependent noise
        self.mean_actions = None  # Mean actions (initialized later)
        self.log_std = None  # Log standard deviation (initialized later)
        self.weights_dist = None  # Distribution of weights used for exploration
        self.exploration_mat = None  # Exploration matrix for single sample
        self.exploration_matrices = None  # Exploration matrices for multiple samples
        self._latent_sde = None  # Latent state-dependent encoding
        self.use_expln = use_expln  # Flag indicating whether to use exponential scaling
        self.use_full_std = (
            use_full_std  # Flag indicating whether to use full standard deviation
        )
        self.epsilon = gsde_epsilon  # Small epsilon value for numerical stability
        self.should_learn_features = (
            should_learn_features  # Flag indicating whether to learn features
        )
        if squash_output:
            self.squasher = Squasher(
                gsde_epsilon
            )  # Squasher object for output squashing
        else:
            self.squasher = None

    def get_std(self, log_std: t.Tensor) -> t.Tensor:
        """
        Computes the standard deviation from the provided log standard deviation.

        Args:
            log_std (t.Tensor): Logarithm of the standard deviation.

        Returns:
            t.Tensor: Standard deviation tensor.
        """
        if self.use_expln:
            # Exponential scaling of standard deviation
            below_threshold = t.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (t.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Standard deviation without exponential scaling
            std = t.exp(log_std)

        if self.use_full_std:
            return std
        return t.ones(self.latent_sde_dim, self.action_size).to(log_std.device) * std

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

        # Initialize the log standard deviation based on whether full std is used or not
        log_std = (
            t.ones(self.latent_sde_dim, self.action_size)
            if self.use_full_std
            else t.ones(self.latent_sde_dim, 1)
        )
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)

        # Sample weights for exploration based on the log standard deviation
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(
        self, mean_actions: t.Tensor, log_std: t.Tensor, latent_sde: t.Tensor
    ) -> Distribution:
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
        if self.squasher is not None:
            # Apply the squasher's inverse function to obtain Gaussian actions
            gaussian_actions = self.squasher.inverse(actions)
        else:
            gaussian_actions = actions

        # Compute the log probability of the Gaussian actions
        log_prob = self.distribution.log_prob(gaussian_actions)
        log_prob = StateDependentNoiseDistribution.sum_independent_dims(log_prob)

        if self.squasher is not None:
            # Compute the log probability correction term for the squashed actions
            log_prob -= t.sum(
                self.squasher.log_prob_correction(gaussian_actions), dim=1
            )
        return log_prob

    def entropy(self) -> Optional[t.Tensor]:
        """
        Computes the entropy of the probability distribution.
        """
        if self.squasher is not None:
            return None
        return StateDependentNoiseDistribution.sum_independent_dims(
            self.distribution.entropy()
        )

    def sample(self) -> t.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.squasher is not None:
            return self.squasher.forward(actions)
        return actions

    def mode(self) -> t.Tensor:
        """
        Computes the mode of the probability distribution.
        """
        actions = self.distribution.mean
        if self.squasher is not None:
            return self.squasher.forward(actions)
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

    @staticmethod
    def sum_independent_dims(tensor: t.Tensor) -> t.Tensor:
        """
        Computes the sum of tensor elements along independent dimensions.

        Args:
            tensor (t.Tensor): Input tensor.

        Returns:
            t.Tensor: Tensor with the sum of elements along independent dimensions.
        """
        if len(tensor.shape) > 1:
            # Sum tensor elements along dimension 1
            tensor = tensor.sum(dim=1)
        else:
            tensor = tensor.sum()
        return tensor
