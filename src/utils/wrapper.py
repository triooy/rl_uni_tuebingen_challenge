import os
import random
from typing import Union
import time

import gym
import laserhockey.hockey_env as lh
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3
import logging
from enum import Enum

from src.HER.utils import (
    compute_reward_classic,
    compute_reward_weighted_distance,
    compute_reward_distance,
    compute_reward_weighted_to_classic,
)

logger = logging.getLogger(__name__)


class Reward(Enum):
    DEFAULT = 0
    POSITIVE = 1
    END = 2


class CustomWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    continuous = False
    NORMAL = 0
    TRAIN_SHOOTING = 1
    TRAIN_DEFENSE = 2

    def __init__(
        self,
        env,
        mode,
        discrete_action_space=False,
        rank=0,
        weak=True,
        reward=Reward.DEFAULT,
        dict_observation_space=False,
        her_reward_function="classic",
    ):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        self.rank = rank
        self.mode = mode
        self.discrete_action_space = discrete_action_space
        if isinstance(reward, int):
            self.reward = Reward(reward)
        else:
            self.reward = reward
        self.opponent = None
        self.opponents = {}
        self.weak = weak
        self.dict_observation_space = dict_observation_space
        self.her_reward_function = her_reward_function
        self.info = {}
        if discrete_action_space:
            env.action_space = spaces.Discrete(7)  # Check if this is still right
        else:
            env.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)
            self.continuous = True
        if mode == self.NORMAL:
            player1 = lh.BasicOpponent(weak=weak)
            self.opponent = player1
        if self.dict_observation_space:
            obs_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(18,)),
                    "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(18,)),
                    "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(18,)),
                }
            )
            self.env.observation_space = obs_space
            self.observation_space = obs_space
            self.env.compute_reward = self.compute_reward

    def convert_obs_to_dict(self, obs):
        obs_tmp = {
            "observation": obs,
            "desired_goal": obs,
            "achieved_goal": obs,
        }
        return obs_tmp

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset()
        if self.dict_observation_space:
            # transform obs to dict for her
            obs = self.convert_obs_to_dict(obs)
        self.info = info
        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """

        if self.mode == self.NORMAL:
            obs_agent2 = self.env.obs_agent_two()
            if isinstance(self.opponent, lh.BasicOpponent):
                a2 = self.opponent.act(obs_agent2)
            else:
                a2 = self.opponent.predict(obs_agent2, deterministic=True)[0]
        else:
            a2 = np.zeros(4)

        if self.discrete_action_space:
            action = self.env.discrete_to_continous_action(action)

        obs, r, d, t, info = self.env.step(np.hstack([action, a2]))

        if self.dict_observation_space:
            obs = self.convert_obs_to_dict(obs)

        info["TimeLimit.truncated"] = d
        info["terminal_observation"] = obs
        r = self._compute_reward(r, info, d)
        if d:
            info["episode"] = {"r": r, "l": self.env.time, "t": d}

        self.info = info
        return obs, r, d, t, info

    def _compute_reward(self, reward, info, done):
        if self.reward == Reward.DEFAULT:
            return reward
        elif self.reward == Reward.POSITIVE:
            return max(reward, 0)
        elif self.reward == Reward.END:
            if done:
                if info["winner"] == 1:
                    return 10
                elif info["winner"] == 0:
                    return 0
                else:
                    return -10
            else:
                return 0

    """
    def compute_reward(self, achieved_goal, desired_goal, info):
        
        rew = np.array([10 * info[i]["winner"] for i in range(achieved_goal.shape[0])])
        return rew
    """
    # 0  x pos player one
    # 1  y pos player one
    # 2  angle player one
    # 3  x vel player one
    # 4  y vel player one
    # 5  angular vel player one
    # 6  x player two
    # 7  y player two
    # 8  angle player two
    # 9 y vel player two
    # 10 y vel player two
    # 11 angular vel player two
    # 12 x pos puck
    # 13 y pos puck
    # 14 x vel puck
    # 15 y vel puck
    # Keep Puck Mode
    # 16 time left player has puck
    # 17 time left other player has puck
    def compute_reward(
        self,
        achieved_goal,
        desired_goal,
        info,
        weights=None,
        p=0.5,
        prob_to_classic=0.01,
    ):
        if self.her_reward_function == "classic":
            return compute_reward_classic(achieved_goal, desired_goal, info)
        elif self.her_reward_function == "distance":
            return compute_reward_distance(achieved_goal, desired_goal, info)
        elif self.her_reward_function == "weighted_distance":
            return compute_reward_weighted_distance(
                achieved_goal, desired_goal, info, weights, p
            )
        elif self.her_reward_function == "weighted_to_classic":
            return compute_reward_weighted_to_classic(
                achieved_goal, desired_goal, info, weights, p, prob_to_classic
            )
        else:
            raise NotImplementedError()

    def set_opponent(self, opponents: Union[list, str, lh.BasicOpponent]):
        if isinstance(opponents, type(lh.BasicOpponent())):
            self.opponent = opponents
            self.opponent.weak = self.weak
            return True
        elif isinstance(opponents, list):
            self.opponent = opponents[self.rank]
        elif isinstance(opponents, str):
            self.opponent = opponents

        if self.mode == self.NORMAL:
            if isinstance(self.opponent, str):
                logger.info(f"Load opponent {self.opponent}")
                if self.opponent in self.opponents.keys():
                    self.opponent = self.opponents[self.opponent]
                    logger.info(f"Loaded opponent {self.opponent}")

                else:
                    opponent = CustomWrapper.load_model_from_disk(path=self.opponent)
                    self.opponents[self.opponent] = opponent
                    self.opponent = opponent

    @staticmethod
    def load_model_from_disk(
        path,
    ):
        # laod opponent from disk
        time.sleep(random.random() * 2)
        # read base dir
        # check if path is zip file or dir
        if os.path.isdir(path):
            files = os.listdir(path)
            model = [file for file in files if file.endswith(".zip")][0]
            path = os.path.join(path, model)
        else:
            files = os.listdir(os.path.dirname(path))
        file = [file for file in files if file.endswith(".txt")][0]
        if file in ["PPO", "PPO.txt"]:
            opponent = ModelWrapperPPO.load(path, device="cpu")
        elif file in ["TD3", "TD3.txt"]:
            opponent = ModelWrapperTD3.load(path, device="cpu")
        else:
            logger.info("Could not load opponent")
            opponent = lh.BasicOpponent()

        logger.info(f"Loaded opponent {opponent}")
        # check if normalize vec env is in dir
        dir = os.path.dirname(path)
        op_name = os.path.basename(path).replace(".zip", "")
        files = os.listdir(dir)
        env = [file for file in files if file.endswith(".pkl")]
        env = [file for file in env if op_name in file]

        # check if normalize vec env is in dir
        if len(env) > 0:
            logger.info(f"Load opponent env {os.path.join(dir, env[0])}")
            opponent.load_env(path=os.path.join(dir, env[0]))
            logger.info(f"Loaded opponent env {os.path.join(dir, env[0])}")

        return opponent


def make_env(
    rank,
    mode=None,
    seed=0,
    discrete_action_space=True,
    reward=Reward.DEFAULT,
    env_weights=[96, 2, 2],
    weak=None,
    dict_observation_space=False,
    her_reward_function="classic",
):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        if mode is None:
            new_mode = random.choices(
                [
                    CustomWrapper.NORMAL,
                    CustomWrapper.TRAIN_SHOOTING,
                    CustomWrapper.TRAIN_DEFENSE,
                ],
                weights=env_weights,
                k=1,
            )[0]
        else:
            new_mode = mode
        if weak is None:
            weak_ = random.choice([True, False])
        else:
            weak_ = weak
        env = lh.HockeyEnv(mode=new_mode)
        """ NEW """

        """ NEW """
        cenv = CustomWrapper(
            env,
            new_mode,
            discrete_action_space,
            reward=reward,
            rank=rank,
            weak=weak_,
            dict_observation_space=dict_observation_space,
            her_reward_function=her_reward_function,
        )
        filename = "monitor.csv"
        cenv = Monitor(cenv, filename=filename)
        return cenv

    set_random_seed(seed)
    return _init


def get_env(
    n_envs,
    mode=None,
    seed=0,
    discrete_action_space=True,
    reward=Reward.DEFAULT,
    env_weights=[96, 2, 2],
    weak=None,
    start_method="fork",
    dict_observation_space=False,
    her_reward_function="classic",
):
    logger.info(
        f"Creating {n_envs} environments, with mode {mode}, \
            seed {seed}, weak {weak}, discrete_action_space {discrete_action_space}, \
                reward {reward}, env_weights {env_weights}"
    )
    env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=mode,
                seed=seed,
                discrete_action_space=discrete_action_space,
                reward=reward,
                weak=weak,  # strength of the basic opponent
                env_weights=env_weights,
                dict_observation_space=dict_observation_space,
                her_reward_function=her_reward_function,
            )
            for i in range(n_envs)
        ],
        start_method=start_method,
    )
    return env


class ModelWrapperTD3(TD3):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.env = None

    def load_env(self, path):
        train_env = DummyVecEnv(
            [
                make_env(
                    i,
                    mode=None,
                    discrete_action_space=False,
                    reward=Reward.DEFAULT,
                )
                for i in range(1)
            ],
        )
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
        )
        self.env = VecNormalize.load(venv=train_env, load_path=path)
        self.env.training = False

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        if self.env is not None:
            obs = self.env.normalize_obs(obs)
        return super().predict(
            obs, state=state, episode_start=episode_start, deterministic=deterministic
        )


class ModelWrapperPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.env = None

    def load_env(self, path):
        train_env = DummyVecEnv(
            [
                make_env(
                    i, mode=None, discrete_action_space=False, reward=Reward.DEFAULT
                )
                for i in range(1)
            ],
        )
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
        )
        self.env = VecNormalize.load(venv=train_env, load_path=path)
        self.env.training = False

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        if self.env is not None:
            obs = self.env.normalize_obs(obs)
        return super().predict(
            obs, state=state, episode_start=episode_start, deterministic=deterministic
        )
