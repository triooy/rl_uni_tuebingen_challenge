import os
import random
import time

import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from rl_zoo3 import linear_schedule
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.ppo.policies import MlpPolicy as PPO_MlpPolicy
from torch import nn as nn

from src.hyperparameter.hyperparams import *
from src.utils.train_callbacks import SaveEnv, SelfplayCallback, TrialEvalCallback
from src.utils.train_utils import compare_params
from src.utils.wrapper import CustomWrapper, make_env


N_procs = 128  # parallel processes (environments)
FILENAME = "/home/dh/logdir/hypersearch_ppo_selfplay3.csv"

STUDY_NAME = "ppo16"  # study for hocky env good run "ppo12"  # optuna study name
TENSORBOARD_LOG_DIR = "/home/dh/logdir/selfplay3/"
STORAGE = "sqlite:////home/dh/ppo/ppo.db"  # optuna storage
N_TRIALS = 200  # number of trials for optuna study
N_STARTUP_TRIALS = 5  # number of trials to be evaluated without pruning
N_EVALUATIONS = 30  # number of evaluations per trial
N_TIMESTEPS = 30_000_000  # number of time steps sum over all environments 
EVAL_FREQ = int(
    N_TIMESTEPS / (N_EVALUATIONS * N_procs)
)  # evaluation frequency in time steps
N_EVAL_EPISODES = 500  # number of episodes for evaluation
STEPS_FOR_TRESHOLD = int(
    8_000_100 / N_procs
)  # if mean reward is below threshold for this number of steps, training is stopped
REWARD_THRESHOLD = 2  # threshold for mean reward
STEPS_FOR_TRESHOLD_2 = int(4_000_100 / N_procs)  # second abort criterion
REWARD_THRESHOLD_2 = 0


## Selfplay parameters
SELF_PLAY = True
ADD_OPPONENT_EVERY_N_STEPS = int(1_000_000 / N_procs)
CHANGE_OPPONENT_EVERY_N_STEPS = int(
    128000 / N_procs
)  # after how many steps the opponent is changed in every environment
FIRST_OPPONENT_AFTER_N_STEPS = int(
    2_000_000 / N_procs
)  # after how many steps the first copy of the agent is added as opponent
HOW_MANY_TO_ADD = 1 # how many copies of the agent are added as opponent
SELFPLAY_EVALUATION = False # takes so much time, that it is not worth it

def objective(trial: optuna.Trial) -> float:
    """
    This function is called by optuna to evaluate a trial.
    Gets a trial object and returns a float value representing the score.
    Runs the training and evaluation of the model.
    """

    start_time = time.time()
    save_env_callback = None
    selfplay_callback = None
    nan_encountered = False
    write = False  # write results to csv file

    # sample hyperparameters
    params = sample_new_ppo_params2(trial)
    # check if hyperparameters are similar to a previous run
    similar = False
    # check if file exists
    if os.path.isfile(FILENAME):
        data = pd.read_csv(FILENAME, index_col=0, header=0)
        similar, reward = compare_params(params, data)
        if similar:
            return reward

    # get environment parameters
    normalize = params.pop("normalize")
    negativ_reward = params.pop("negative_reward")
    discrete_action_space = params.pop("discrete_action_space")
    print("------------------------------")
    print("Training with the following params:")
    print(params)
    print(f"normalize: {normalize}")
    print(f"negative_reward: {negativ_reward}")
    print(f"discrete_action_space: {discrete_action_space}")
    print("------------------------------")

    # create environments
    train_env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=None,
                discrete_action_space=discrete_action_space,
                negativ_reward=negativ_reward,
                weak=False,  # strength of the basic opponent
            )
            for i in range(N_procs)
        ],
        start_method="fork",
    )
    # create evaluation environment
    eval_env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=CustomWrapper.NORMAL,
                discrete_action_space=discrete_action_space,
                negativ_reward=True,
                weak=False,
            )
            for i in range(10)  # 10 evaluation environments
            # random number
            # for selfplay it does matter, because for each env a random selfplay agent is added as opponent
            # larger the better
        ],
        start_method="fork",
    )
    # normalize input and reward
    if normalize:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # reward is not normalized for evaluation
            training=False,  # training is set to false, so that the running mean and std are not updated
        )

    # reset environments
    train_env.reset()
    eval_env.reset()

    # complete parameters so that they can be used for training
    params["env"] = train_env
    params["policy"] = PPO_MlpPolicy
    lr_schedule = params.pop("lr_schedule")
    lr_rate = params["learning_rate"]
    if lr_schedule == "linear":
        params["learning_rate"] = linear_schedule(lr_rate)

    
    
    print(params)
    model = PPO(
        **params,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
            ortho_init=True,
        ),
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

    ## Callbacks
    # Setup callbacks

    # Stop training if there is no improvement after more than 5 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, min_evals=6, verbose=1
    )
    if normalize:
        # Save the environment that was used for training
        # This is important because the running mean and std are not saved in the model
        save_env_callback = SaveEnv(
            path=os.path.join(
                TENSORBOARD_LOG_DIR, f"trial_{trial.number}/best_model_env.pkl"
            )
        )

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env,  # evaluation environment
        trial,  # optuna trial (for reporting)
        best_model_save_path=os.path.join(TENSORBOARD_LOG_DIR, f"trial_{trial.number}"),
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        callback_after_eval=stop_train_callback,
        callback_on_new_best=save_env_callback,
        verbose=1,
        reward_threshold=REWARD_THRESHOLD,
        reward_threshold_2=REWARD_THRESHOLD_2,
        steps_for_threshold_2=STEPS_FOR_TRESHOLD_2,
        steps_for_threshold=STEPS_FOR_TRESHOLD,
        selfplay_evaluation=SELFPLAY_EVALUATION,
    )

    callbacks = [eval_callback]

    if SELF_PLAY:
        # Create the callback that will periodically add a copy of the agent as opponent
        # and update the opponent every n steps
        selfplay_callback = SelfplayCallback(
            verbose=1,
            add_opponent_every_n_steps=ADD_OPPONENT_EVERY_N_STEPS,
            change_every_n_steps=CHANGE_OPPONENT_EVERY_N_STEPS,
            save_path=os.path.join(TENSORBOARD_LOG_DIR, f"trial_{trial.number}"),
            n_envs=N_procs,
            first_opponent_after_n_steps=FIRST_OPPONENT_AFTER_N_STEPS,
            how_many_to_add=HOW_MANY_TO_ADD,
        )

        eval_callback.selfplay_callback = selfplay_callback  # add selfplay callback to evaluation callback to evaluate against the selfplay agent
        callbacks += [selfplay_callback]  # add selfplay callback to training callbacks

    ## Training

    try:
        # Train the model for N_TIMESTEPS
        model.learn(
            N_TIMESTEPS,
            callback=callbacks,  # evaluation and selfplay callback
            tb_log_name=f"trial_{STUDY_NAME}_" + str(trial.number),
        )
        # Evaluate the final model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=500)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # Evaluate the selfplay agent
        if selfplay_callback is not None and len(selfplay_callback.opponents) > 1:
            # evaluate against the last selfplay agent
            new_opponents = selfplay_callback.opponents[1:][-eval_env.num_envs:]
            if len(new_opponents) < eval_env.num_envs:
                new_opponents += [selfplay_callback.opponents[-1] for i in range(eval_env.num_envs - len(new_opponents))]
            eval_env.env_method(
                "set_opponent", new_opponents
            )  # set opponents for evaluation environments
            # evaluate against the selfplay agent
            selfplay_mean_reward, selfplay_std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=1000
            )
            print(
                f"selfplay_mean_reward:{selfplay_mean_reward:.2f} +/- {selfplay_std_reward:.2f}"
            )
        else:
            selfplay_mean_reward = -np.inf
            selfplay_std_reward = 0
        write = True
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    except ValueError:
        nan_encountered = True
        mean_reward = -1000
        std_reward = 0
    finally:
        end_time = time.time()
        training_time = end_time - start_time
        # in minutes
        training_time /= 60
        print(f"Training took {training_time} min")
        # create dataframe with hyperparameters and results
        params.pop("policy")
        params.pop("env")
        params["lr_schedule"] = lr_schedule
        params["learning_rate"] = lr_rate
        params["trial_number"] = trial.number
        params["device"] = torch.cuda.get_device_name(0)
        params["steps"] = model.num_timesteps

        if write:
            data = {
                "params": [str(params)],
                "normalize": [normalize],
                "negativ_reward": [negativ_reward],
                "discrete_action_space": [discrete_action_space],
                "training_time": [training_time],
                "mean_reward": [mean_reward],
                "std_reward": [std_reward],
            }
            if SELF_PLAY:
                data["selfplay_mean_reward"] = [selfplay_mean_reward]
                data["selfplay_std_reward"] = [selfplay_std_reward]
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame(data)
            # check if file exists
            if os.path.isfile(FILENAME):
                df.to_csv(FILENAME, mode="a", header=False)
            else:
                df.to_csv(FILENAME, mode="a", header=True)

        # Free memory.
        model.env.close()
        eval_env.close()

        del model

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training. ?
    # torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE, 
        load_if_exists=True,
    )
    study.enqueue_trial(    
        {
        "n_steps": 128,
        "batch_size": 2048,
        "gamma": 0.995,
        "learning_rate": 0.00085,
        "lr_schedule": 'constant',
        "ent_coef": 0.00015525584920320315,
        "clip_range": 0.3,
        "n_epochs": 5,
        "gae_lambda": 0.92,
        "max_grad_norm": 0.3,
        "vf_coef": 0.2422495988381274,
        "normalize": False,
        "negative_reward": True,
        "discrete_action_space": False,
    })
    
    
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
