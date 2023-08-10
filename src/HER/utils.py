import numpy as np

""" By Cornelius Wiehl, 10.8.2023 
Different Reward Computing Functions used during Hindisght Experience Replay
These functions are actually used to overwrite the compute_reward method of the environment.
They are imported in the src/utils/wrapper.py and then used in the HER class.
"""


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
    achieved_goal, desired_goal, info, weights=None, percentage_to_classic=0.01
):
    p = 0.5
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
    achieved_goal,
    desired_goal,
    info,
    weights=None,
):
    p = 0.5
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
