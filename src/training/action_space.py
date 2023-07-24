import logging
import time
import pandas as pd
import numpy as np
import os
from typing import Optional
from src.utils.wrapper import get_env, CustomWrapper, Reward
from stable_baselines3.common.evaluation import evaluate_policy


logger = logging.getLogger(__name__)


def get_action_space_params(config, model1_dir, weak_opponent, model2_dir=None):
    logger.info("Start Action Space Evaluation...")
    DISCRETE_ACTION_SPACE = config["action_space"]["discrete_action_space"]

    model1 = model1_dir
    model1 = CustomWrapper.load_model_from_disk(model1)
    eval_env = get_env(
        mode=CustomWrapper.NORMAL,
        discrete_action_space=DISCRETE_ACTION_SPACE,
        reward=Reward.END,
        weak=config["action_space"]["weak_opponent"],
        n_envs=config["action_space"]["n_eval_envs"],
        start_method=config["action_space"]["start_method"],
    )
    eval_env.render_mode = config["action_space"]["render_mode"]

    if model2_dir is not None:
        model2 = model2_dir
        eval_env.env_method("set_opponent", model2)

    obs = eval_env.reset()

    render = config["action_space"]["render"]

    action_space = []
    observation_space = []
    reward_space = []
    for i in range(1000):
        for _ in range(10000):
            if render:
                eval_env.render()
            a1 = model1.predict(obs, deterministic=True)[0]
            action_space.append(a1[0])
            obs, r, d, info = eval_env.step(a1)
            observation_space.append(obs[0])
            reward_space.append(r[0])
            if any(d):
                break
            if render:
                time.sleep(0.02)
        obs = eval_env.reset()

    actions = list(np.vstack(action_space).astype(float))
    observations = list(np.vstack(observation_space).astype(float))
    rewards = list(np.vstack(reward_space).astype(float))
    return actions, observations, rewards


def action_space(config):
    models = config["action_space"]["models"]
    opponents = config["action_space"]["opponents"]
    weak_opponent = config["action_space"]["weak_opponent"]

    df = pd.DataFrame(
        columns=[
            "action_space",
            "observation_space",
            "reward_space",
            "model",
            "opponent",
        ],
        index=np.arange(len(models)),
    )
    i = 0
    for model, opponent in zip(models, opponents):

        if opponent == "weak":
            weak_opponent = True
            model2_dir = None
        elif opponent == "strong":
            weak_opponent = False
            model2_dir = None
        else:
            model2_dir = opponent

        logger.info("Start Action Space Evaluation...")
        logger.info("Model: " + model)
        logger.info("Opponent: " + opponent)
        action, observations, rewards = get_action_space_params(
            config, model, weak_opponent, model2_dir=model2_dir
        )

        df.loc[i, "action_space"] = action
        df.loc[i, "observation_space"] = observations
        df.loc[i, "reward_space"] = rewards
        df.loc[i, "model"] = model
        df.loc[i, "opponent"] = opponent
        i += 1

    save_path = config["action_space"]["save_location"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, "action_space.pkl")
    df.to_pickle(file_path)
