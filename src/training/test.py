import logging
import time
from typing import Optional
from src.utils.wrapper import get_env, CustomWrapper, Reward
from stable_baselines3.common.evaluation import evaluate_policy


logger = logging.getLogger(__name__)


def test(config):
    logger.info("Start testing...")
    DISCRETE_ACTION_SPACE = config["test"]["discrete_action_space"]
    if "dict_observation_space" in config["test"]:
        dict_observation_space = config["test"]["dict_observation_space"]
    else:
        dict_observation_space = False
    model1 = config["test"]["model_1_dir"]
    model1 = CustomWrapper.load_model_from_disk(
        model1, dict_observation_space=dict_observation_space
    )
    eval_env = get_env(
        mode=CustomWrapper.NORMAL,
        discrete_action_space=DISCRETE_ACTION_SPACE,
        reward=Reward.END,
        weak=config["test"]["weak_opponent"],
        n_envs=config["test"]["n_eval_envs"],
        start_method=config["test"]["start_method"],
        dict_observation_space=config["test"]["dict_observation_space"],
    )
    eval_env.render_mode = config["test"]["render_mode"]

    if config["test"]["model_vs_model"]:
        model2 = config["test"]["model_2_dir"]
        eval_env.env_method("set_opponent", model2)

    obs = eval_env.reset()

    mean_reward, std_reward = evaluate_policy(
        model1,
        eval_env,
        n_eval_episodes=config["test"]["n_eval_episodes"],
        deterministic=True,
    )
    logger.info(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    for i in range(1000):
        for _ in range(10000):
            eval_env.render()
            a1 = model1.predict(obs, deterministic=True)[0]
            obs, r, d, info = eval_env.step(a1)
            if any(d):
                break
            time.sleep(0.02)
        obs = eval_env.reset()
