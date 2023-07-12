from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import optuna
import numpy as np
import os
from src.training.train import train
from src.hyperparameter.hyperparams import (
    sample_new_ppo_params2,
    sample_td3_params,
    sample_gsde_ppo_params,
)
import logging

logger = logging.getLogger(__name__)


def get_objective_fn(config):
    """
    This function is called by optuna to evaluate a trial.
    Gets a trial object and returns a float value representing the score.
    Runs the training and evaluation of the model.
    """
    run_name = config["logs"]["run_name"]
    if config["agent"]["agent"] == "PPO":
        sample_fn = sample_gsde_ppo_params
    elif config["agent"]["agent"] == "TD3":
        sample_fn = sample_td3_params
    else:
        raise NotImplementedError

    def objective(trial: optuna.Trial) -> float:
        try:
            params = sample_fn(trial)
            normalize = params.pop("normalize")
            negative_reward = params.pop("negative_reward")
            discrete_action_space = params.pop("discrete_action_space")
            config["agent"]["normalize"] = normalize
            config["agent"]["negative_reward"] = negative_reward
            config["agent"]["discrete_action_space"] = discrete_action_space
            config["agent_parameter"] = params
            config["logs"]["run_name"] = run_name + f"_{trial.number}"
            trainer = train(config)
            path = os.path.join(
                config["logs"]["tensorboard_log_dir"],
                config["hyperparameter"]["csv_filename"],
            )
            trainer.write_csv(path)
            return trainer.mean_reward
        except Exception as e:
            logger.error(e, exc_info=True)
            return -np.inf

    return objective


def hyperparameter_search(config):
    hyperparameter = config["hyperparameter"]
    training = config["training"]

    sampler = TPESampler(n_startup_trials=hyperparameter["n_startup_trials"])
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=hyperparameter["n_startup_trials"],
        n_warmup_steps=training["n_evaluations"] // 3,
    )

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name=hyperparameter["study_name"],
        storage=hyperparameter["storage"],
        load_if_exists=True,
    )
    objective = get_objective_fn(config)
    try:
        study.optimize(
            objective,
            n_trials=hyperparameter["n_trials"],
            n_jobs=hyperparameter["n_jobs"],
        )
    except KeyboardInterrupt:
        pass

    logger.info("Number of finished trials: ", len(study.trials))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: ", trial.value)

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    logger.info("  User attrs:")
    for key, value in trial.user_attrs.items():
        logger.info("    {}: {}".format(key, value))
