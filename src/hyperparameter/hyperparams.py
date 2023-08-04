from typing import Any, Dict

import numpy as np
import optuna
from rl_zoo3 import linear_schedule
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from torch import nn as nn
from collections import defaultdict


def sample_new_ppo_params2(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = 2048  # trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192, 16384, 32768])
    n_steps = (
        512  # trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096, 8192])
    )
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, step=1e-5)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', [True, False])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.001, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = 5  # trial.suggest_categorical("n_epochs", [1, 3, 5,])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # net_arch = 'medium'
    normalize = True  # trial.suggest_categorical("normalize", [True, False])
    reward = trial.suggest_categorical("reward", [0, 2])
    discrete_action_space = (
        False  # trial.suggest_categorical("discrete_action_space", [True, False])
    )
    policy = trial.suggest_categorical(
        "policy", ["PPO_gSDE_MlpPolicy", "PPO_MlpPolicy"]
    )
    if policy == "PPO_gSDE_MlpPolicy":
        use_gSDE = True
        log_std_init = -2.5989870268124142
    else:
        use_gSDE = False
        log_std_init = 0

    # Orthogonal initialization
    # ortho_init = True
    # activation_fn = trial.suggest_categorical(
    #    "activation_fn", ["relu", "tanh", "leaky_relu"]
    # )
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # activation_fn = "relu"

    # TODO: account when using multiple envs
    if batch_size >= (n_steps * 128):  # 128 envs
        batch_size = n_steps * 128

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    # net_arch = {
    #    "small": dict(pi=[64, 64], vf=[64, 64]),
    #    "medium": dict(pi=[256, 256], vf=[256, 256]),
    # }[net_arch]

    # activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh,  "leaky_relu": nn.LeakyReLU}[
    #     activation_fn
    # ]

    # selfplay
    how_many_to_add = trial.suggest_categorical("how_many_to_add", [0, 1, 2, 5, 10])

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "normalize": normalize,
        "reward": reward,
        "use_sde": use_gSDE,
        "discrete_action_space": discrete_action_space,
        "policy_kwargs": dict(
            net_arch=[256, 256],
            ortho_init=True,
            activation_fn="relu",
            log_std_init=log_std_init,
        ),
        "policy": policy,
        "how_many_to_add": how_many_to_add,
        "sde_sample_freq": 16,
    }


def sample_gsde_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    sde_sample_freq = trial.suggest_categorical(
        "sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256]
    )
    use_gSDE = True
    should_learn_features = trial.suggest_categorical(
        "should_learn_features", [True, False]
    )

    return {
        "n_steps": 512,
        "batch_size": 2048,
        "gamma": 0.999,
        "learning_rate": 0.00073,
        "ent_coef": 0.0009673352,
        "clip_range": 0.2,
        "n_epochs": 5,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.6,
        "vf_coef": 0.6482388887006054,
        "normalize": True,
        "negative_reward": True,
        "discrete_action_space": False,
        "use_sde": use_gSDE,
        "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            net_arch=[256, 256],
            ortho_init=True,
            activation_fn="relu",
            log_std_init=log_std_init,
            should_learn_features=should_learn_features,
        ),
    }


def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, step=1e-5
    )  # 1e-5=0.00001
    batch_size = (
        2048  # trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    )
    buffer_size = 1000000  # trial.suggest_categorical("buffer_size", [int(1e5), int(1e6), int(1e7)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical("train_freq", [256, 512, 1024, 2048, 4096])
    gradient_steps = train_freq

    noise_type = trial.suggest_categorical(
        "noise_type", ["ornstein-uhlenbeck", "normal", None]
    )
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    normalize = True  # trial.suggest_categorical("normalize", [True, False])
    reward = trial.suggest_categorical("reward", [0, 2])
    discrete_action_space = (
        False  # trial.suggest_categorical("discrete_action_space", [True, False])
    )

    # NOTE: Add "verybig" to net_arch when tuning HER
    # net_arch = trial.suggest_categorical(
    #     "net_arch",
    #     [
    #         # "small",
    #         "medium",
    #         # "big",
    #         "verybig",
    #     ],
    # )
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
    # ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    # net_arch = {
    #     # "small": [64, 64],
    #     "medium": [256, 256],
    #     # "big": [400, 300],
    #     # Uncomment for tuning HER
    #     "verybig": [256, 256, 256],
    # }[net_arch]

    policy_delay = trial.suggest_categorical("policy_delay", [1, 2, 4])
    target_noise_clip = trial.suggest_categorical("target_noise_clip", [0.1, 0.2, 0.3])

    # hindsight_replay_buffer = trial.suggest_categorical("her", [True, False])
    # if hindsight_replay_buffer:
    #     dict_observation_space = True
    #     policy = "TD3_MultiInputPolicy"
    # else:
    #     dict_observation_space = False
    #     policy = "TD3_MlpPolicy"

    # selfplay
    how_many_to_add = trial.suggest_categorical("how_many_to_add", [0, 1, 2, 5, 10])

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        #  "policy_kwargs": dict(net_arch=net_arch),
        "tau": tau,
        "policy_delay": policy_delay,
        "target_noise_clip": target_noise_clip,
        "how_many_to_add": how_many_to_add,
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(4), sigma=noise_std * np.ones(4)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(4), sigma=noise_std * np.ones(4)
        )

    hyperparams["normalize"] = normalize
    hyperparams["reward"] = reward
    hyperparams["discrete_action_space"] = discrete_action_space

    return defaultdict(lambda: False, hyperparams)


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    )
    n_steps = trial.suggest_categorical(
        "n_steps", [128, 256, 512, 1024, 2048, 4096, 8192]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical(
        "n_epochs",
        [
            1,
            3,
            5,
        ],
    )
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    normalize = trial.suggest_categorical("normalize", [True, False])
    reward = trial.suggest_categorical("reward", [0, 1, 2])
    discrete_action_space = trial.suggest_categorical("discrete_action_space", [False])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > (n_steps * 128):  # 128 envs
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
        "normalize": normalize,
        "reward": reward,
        "discrete_action_space": discrete_action_space,
    }
