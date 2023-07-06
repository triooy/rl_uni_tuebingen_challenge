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

logger = logging.getLogger(__name__)


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
        negativ_reward=True,
        rank=0,
        weak=True,
    ):
        # Call the parent constructor, so we can access self.env later
        super().__init__(env)
        self.rank = rank
        self.mode = mode
        self.discrete_action_space = discrete_action_space
        self.negativ_reward = negativ_reward
        self.opponent = None
        self.opponents = {}
        self.weak = weak
        if discrete_action_space:
            env.action_space = spaces.Discrete(7)  # Check if this is still right
        else:
            env.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)
            self.continuous = True
        if mode == self.NORMAL:
            player1 = lh.BasicOpponent(weak=weak)
            self.opponent = player1

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset()

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
        info["TimeLimit.truncated"] = d
        info["terminal_observation"] = obs
        if not self.negativ_reward:
            r = max(r, 0)
        if d:
            d = True
            info["episode"] = {"r": r, "l": self.env.time, "t": d}
        return obs, r, d, t, info

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
    negativ_reward=False,
    env_weights=[96, 2, 2],
    weak=None,
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
        cenv = CustomWrapper(
            env, new_mode, discrete_action_space, negativ_reward, rank=rank, weak=weak_
        )
        cenv = Monitor(cenv, filename=None)
        return cenv

    set_random_seed(seed)
    return _init


def get_env(
    n_envs,
    mode=None,
    seed=0,
    discrete_action_space=True,
    negative_reward=False,
    env_weights=[96, 2, 2],
    weak=None,
    start_method="fork",
):
    env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=mode,
                seed=seed,
                discrete_action_space=discrete_action_space,
                negativ_reward=negative_reward,
                weak=weak,  # strength of the basic opponent
                env_weights=env_weights,
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
                    i, mode=None, discrete_action_space=False, negativ_reward=False
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
                    i, mode=None, discrete_action_space=False, negativ_reward=False
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
