from gymnasium import spaces
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np


""" Different Reward Computing Functions"""


def compute_reward_classic(achieved_goal, desired_goal, info):
    """Resembles the reward function of the environment."""
    winners = np.array([w["winner"] for w in info])
    winners_idx = np.where(winners == 1)[0]
    loosers_idx = np.where(winners == -1)[0]
    rew = np.zeros(achieved_goal.shape[0])
    rew[winners_idx] = 10
    rew[loosers_idx] = -10
    return rew


def compute_reward_weighted_to_classic(
    achieved_goal, desired_goal, info, weights=None, p=0.5, percentage_to_classic=0.05
):
    if weights is None:
        weights = [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.1,
            0.1,
            0.1,
            0.1,
            0,
            0,
        ]
    winners = np.array([w["winner"] for w in info])
    winners_idx = np.where(winners == 1)[0]
    if len(winners_idx) > int(percentage_to_classic * achieved_goal.shape[0]):
        rew = compute_reward_classic(achieved_goal, desired_goal, info)
    else:
        rew = compute_reward_classic(achieved_goal, desired_goal, info)
    return rew


def compute_reward_distance(achieved_goal, desired_goal, info):
    rew = np.sum(np.abs(achieved_goal - desired_goal), axis=1)
    return -rew


def compute_reward_weighted_distance(
    achieved_goal, desired_goal, info, weights=None, p=0.5
):
    if weights is None:
        weights = [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0.1,
            0.1,
            0.1,
            0.1,
            0,
            0,
        ]
    dist = np.abs(achieved_goal - desired_goal)
    rew = np.dot(dist, weights)
    rew = -np.power(rew, p)
    return rew


def compute_reward_old(self, achieved_goal, desired_goal, info):
    p = 0.5
    # x = 1 / 18
    """
        weigths = np.array(
            [
                0.001,
                0.001,
                0.001,
                0.001,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.001,
                0.001,
                0.001,
                0.001,
                0,
                0,
            ]
        )
        
        weigths = np.array(
            [
                0.125,  # 0.125,
                0.125,
                0.125,
                0.125,
                0.125,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.125,  # 0.125,
                0.125,  # 0.125,
                0.125,  # 0.125,
                0.125,  # 0.125,
                0,
                0,
            ]
        )
        # res = -np.power(np.dot(np.abs(achieved_goal - desired_goal), weigths), p)
        # res = -np.dot(np.abs(achieved_goal - desired_goal), weigths)

        # do not block reward once the goal is achieved
        winners = np.array([w["winner"] for w in info])
        winners_idx = np.where(winners == 1)[0]
        loosers_idx = np.where(winners == -1)[0]

        if len(winners_idx) > 0 or True:  # int(0.01 * len(achieved_goal)):
            # weigths[:5] = np.array([0, 0, 0, 0, 0])
            weigths = np.zeros(18)

        # res = -np.dot(np.abs(achieved_goal - desired_goal), weigths)
        res = -np.dot(
            np.abs(achieved_goal[:, [12, 13]] - achieved_goal[:, [0, 1]]),
            weigths[12:14],
        )

        res[winners_idx] = 10
        res[loosers_idx] = -10

        # res = np.ones(achieved_goal.shape[0]) * 100
        """
    desired_goal = np.ones(achieved_goal.shape[0])
    winners = np.array([w["winner"] for w in info])

    winners_idx = np.where(winners == 1)[0]

    res = -10 * np.ones(achieved_goal.shape[0])
    res[winners_idx] = 10

    # res = -(desired_goal - winners) + 1

    return res


""" Other Utils """


def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
