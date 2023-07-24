import torch as t
from typing import Optional


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


def get_mean(distribution) -> t.Tensor:
    """
    Returns the mode of the probability distribution.
    """
    return distribution.mean


def get_entropy(gsde) -> Optional[t.Tensor]:
    """
    Returns the entropy of the probability distribution.
    """
    return sum_independent_dims(gsde.distribution.entropy())
