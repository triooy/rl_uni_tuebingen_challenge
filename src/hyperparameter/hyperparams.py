from typing import Any, Dict

import numpy as np
import optuna
from rl_zoo3 import linear_schedule
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)
from torch import nn as nn


def sample_new_ppo_params3(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = (
        512  #  trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    )
    n_steps = 256  # trial.suggest_categorical("n_steps", [128, 256, 512])
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
    net_arch = "small"
    normalize = trial.suggest_categorical("normalize", [True, False])
    negative_reward = (
        True  # trial.suggest_categorical("negative_reward", [True, False])
    )
    discrete_action_space = (
        False  # trial.suggest_categorical("discrete_action_space", [True, False])
    )

    # Orthogonal initialization
    ortho_init = True
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "leaky_relu"]
    )
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > (n_steps * 128):  # 128 envs
        batch_size = n_steps

    # if lr_schedule == "linear":
    #    learning_rate = linear_schedule(learning_rate)

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
        "lr_schedule": lr_schedule,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        #  "use_sde":use_sde,
        "policy_kwargs": dict(
            #    log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
        "normalize": normalize,
        "negative_reward": negative_reward,
        "discrete_action_space": discrete_action_space,
    }


def sample_new_ppo_params_negativ_reward(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
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
    net_arch = "small"
    normalize = trial.suggest_categorical("normalize", [True, False])
    negative_reward = (
        True  # trial.suggest_categorical("negative_reward", [True, False])
    )
    discrete_action_space = (
        False  # trial.suggest_categorical("discrete_action_space", [True, False])
    )

    # Orthogonal initialization
    ortho_init = True
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
    )
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > (n_steps * 128):  # 128 envs
        batch_size = n_steps

    # if lr_schedule == "linear":
    #    learning_rate = linear_schedule(learning_rate)

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
        "lr_schedule": lr_schedule,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        #  "use_sde":use_sde,
        "policy_kwargs": dict(
            #    log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
        "normalize": normalize,
        "negative_reward": negative_reward,
        "discrete_action_space": discrete_action_space,
    }


def sample_new_ppo_params2(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048, 4096])
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512])
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
    #net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    #net_arch = 'medium'
    normalize = trial.suggest_categorical("normalize", [True, False])
    negative_reward = trial.suggest_categorical("negative_reward", [True, False])
    discrete_action_space = (
        False  # trial.suggest_categorical("discrete_action_space", [True, False])
    )

    # Orthogonal initialization
    #ortho_init = True
    #activation_fn = trial.suggest_categorical(
    #    "activation_fn", ["relu", "tanh", "leaky_relu"]
    #)
    #activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    #activation_fn = "relu"
    
    # TODO: account when using multiple envs
    if batch_size > (n_steps * 128):  # 128 envs
        batch_size = n_steps

    # if lr_schedule == "linear":
    #    learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    #net_arch = {
    #    "small": dict(pi=[64, 64], vf=[64, 64]),
    #    "medium": dict(pi=[256, 256], vf=[256, 256]),
    #}[net_arch]

    # activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh,  "leaky_relu": nn.LeakyReLU}[
    #     activation_fn
    # ]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "normalize": normalize,
        "negative_reward": negative_reward,
        "discrete_action_space": discrete_action_space,
    }


def sample_new_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
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
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    lr_schedule = trial.suggest_categorical("lr_schedule", [True, False])
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
    # net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    net_arch = "medium"
    normalize = trial.suggest_categorical("normalize", [True, False])
    negative_reward = trial.suggest_categorical("negative_reward", [True, False])
    discrete_action_space = trial.suggest_categorical(
        "discrete_action_space", [True, False]
    )
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # use_sde = trial.suggest_categorical("use_sde", [True, False])
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # if not use_sde:
    #    sde_sample_freq = -1
    # Orthogonal initialization
    ortho_init = True
    activation_fn = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
    )
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > (n_steps * 128):  # 128 envs
        batch_size = n_steps

    # if lr_schedule == "linear":
    #    learning_rate = linear_schedule(learning_rate)

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
        "lr_schedule": lr_schedule,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        #  "use_sde":use_sde,
        "policy_kwargs": dict(
            #    log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
        "normalize": normalize,
        "negative_reward": negative_reward,
        "discrete_action_space": discrete_action_space,
    }


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
    negative_reward = trial.suggest_categorical("negative_reward", [True, False])
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
        "negative_reward": negative_reward,
        "discrete_action_space": discrete_action_space,
    }
