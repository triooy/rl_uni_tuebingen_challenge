from stable_baselines3.common.distributions import Distribution
import torch as th
from torch.distributions import Normal
from torch import nn
from typing import Optional, Tuple


class Squasher:
    """
    The Squasher class is used to squash the output of the policy network
    using tanh
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return Squasher.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        # Squash correction (from original SAC implementation)
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)


class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719

    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
        self,
        action_dim: int,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        learn_features: bool = False,
        gsde_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = gsde_epsilon
        self.learn_features = learn_features
        if squash_output:
            self.squasher = Squasher(gsde_epsilon)
        else:
            self.squasher = None

    def get_std(self, log_std: th.Tensor) -> th.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = th.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = th.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return th.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

    def sample_weights(self, log_std: th.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(
        self,
        latent_dim: int,
        log_std_init: float = -2.0,
        latent_sde_dim: Optional[int] = None,
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        # Network for the deterministic action, it represents the mean of the distribution
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = (
            th.ones(self.latent_sde_dim, self.action_dim)
            if self.full_std
            else th.ones(self.latent_sde_dim, 1)
        )
        # Transform it to a parameter so it can be optimized
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> Distribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = th.mm(self._latent_sde**2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.squasher is not None:
            gaussian_actions = self.squasher.inverse(actions)
        else:
            gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        if self.squasher is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= th.sum(
                self.squasher.log_prob_correction(gaussian_actions), dim=1
            )
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        if self.squasher is not None:
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.squasher is not None:
            return self.squasher.forward(actions)
        return actions

    def mode(self) -> th.Tensor:
        actions = self.distribution.mean
        if self.squasher is not None:
            return self.squasher.forward(actions)
        return actions

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat.to(latent_sde.device))
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def actions_from_params(
        self,
        mean_actions: th.Tensor,
        log_std: th.Tensor,
        latent_sde: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor
