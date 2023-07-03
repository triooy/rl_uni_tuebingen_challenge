import os
import random

import gymnasium
import laserhockey.hockey_env as lh
import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


class SelfplayCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        verbose=0,
        change_every_n_steps=1000,
        add_opponent_every_n_steps=1000,
        save_path="./",
        n_envs=128,
        first_opponent_after_n_steps=0,
        how_many_to_add=None, # how many copies of a particular opponent to add to the list e.g.: 128 env add 2 copies of every past model
                            # one model every 1 million steps, training for 20 million steps = 40 models 
                            # None = random 
    ):
        super(SelfplayCallback, self).__init__(verbose)
        self.change_every_n_steps = change_every_n_steps
        self.add_opponent_every_n_steps = add_opponent_every_n_steps
        self.opponents = [lh.BasicOpponent()]
        self.save_path = save_path
        self.n_envs = n_envs
        self.first_opponent_after_n_steps = first_opponent_after_n_steps
        self.how_many_to_add = how_many_to_add

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls > self.first_opponent_after_n_steps:
            if (
                self.add_opponent_every_n_steps > 0
                and self.n_calls % self.add_opponent_every_n_steps == 0
            ):
                # First save the current model
                path = f"{self.save_path}/model_{self.n_calls}.zip"
                self.model.save(path)
                vec_env = self.model.get_vec_normalize_env()
                # if there is a normalization env, save that to
                if vec_env:
                    vec_env.save(
                        os.path.join(self.save_path, f"model_{self.n_calls}.pkl")
                    )
                self.opponents.append(path)
                if self.verbose > 0:
                    print(f"Added self as opponent after {self.n_calls} steps")
            if (
                self.change_every_n_steps > 0
                and self.n_calls % self.change_every_n_steps == 0
            ):  
                if self.how_many_to_add: # how many copies to add
                    new_opponents = []
                    for opponent in self.opponents:
                        new_opponents += [opponent for i in range(self.how_many_to_add)]
                    for i in range(self.n_envs - len(new_opponents)):
                        new_opponents.append(lh.BasicOpponent(weak=False))
                else: #otherwise choose random 
                    new_opponents = random.choices(
                        self.opponents,
                        k=self.n_envs,
                    )
                # set the opponents in the envs
                self.model.get_env().env_method("set_opponent", new_opponents)
                # log some stats
                count_basic_opponent = [
                    isinstance(opponent, lh.BasicOpponent) for opponent in new_opponents
                ].count(True)
                self.logger.record(
                    "eval/number_of_selfs", self.n_envs - count_basic_opponent
                )
                if self.verbose > 0:
                    print(f"Changed opponent for envs after {self.n_calls} steps")
                    print(
                        f"Number of basic opponents: {count_basic_opponent}, number of opponents: {len(self.opponents)}"
                    )
        return True
    
    def add_opponents(self, opponent):
        if isinstance(opponent, str):
            self.opponents.append(opponent)
        elif isinstance(opponent, list):
            self.opponents += opponent
        print(f"Added opponent {opponent} to list of opponents")


class SaveEnv(BaseCallback):
    """
    Saves the environment if there is a new best model.
    """

    def __init__(self, verbose=0, path=None):
        super(SaveEnv, self).__init__(verbose)
        self.path = path

    def _on_step(self) -> bool:
        env = self.model.get_vec_normalize_env()
        if env:
            env.save(self.path)
            if self.verbose > 0:
                print(f"Saved env to {self.path}")
        else:
            print("No VecNormalize env to save")
        return True


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        best_model_save_path: str,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        callback_after_eval=None,
        callback_on_new_best=None,
        n_procs=128,
        reward_threshold=-10,
        reward_threshold_2=-10,
        steps_for_threshold_2=1000000,
        steps_for_threshold=1000000,
        selfplay_callback=None,
        selfplay_evaluation=False,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            callback_after_eval=callback_after_eval,
            best_model_save_path=best_model_save_path,
            callback_on_new_best=callback_on_new_best,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.over_threshold = False
        self.over_threshold_2 = False
        self.n_procs = n_procs
        self.reward_threshold = reward_threshold
        self.reward_threshold_2 = reward_threshold_2
        self.steps_for_threshold_2 = steps_for_threshold_2
        self.steps_for_threshold = steps_for_threshold
        self.selfplay_best_mean_reward = -np.inf
        self.selfplay_last_mean_reward = -np.inf
        self.selfplay_callback = selfplay_callback
        self.selfplay_evaluation = selfplay_evaluation
        
    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            continue_training = super()._on_step() # run normal evaluation

            ### Selfplay
            if self.selfplay_callback and self.selfplay_evaluation:
                self.selfplay_evaluation()

            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        elif (
            self.n_calls < self.steps_for_threshold
            and self.last_mean_reward > self.reward_threshold
            and not self.over_threshold
        ):
            self.over_threshold = True
            print(f"Passed threshold after {self.n_calls * self.n_procs}")
        elif (
            self.n_calls > self.steps_for_threshold
            and self.last_mean_reward < self.reward_threshold
            and not self.over_threshold
        ):
            continue_training = False
            print(
                f"Stop training because reward {self.last_mean_reward} is below {self.reward_threshold} after {self.n_calls * self.n_procs} steps"
            )
        elif (
            self.n_calls < self.steps_for_threshold_2
            and self.last_mean_reward > self.reward_threshold_2
            and not self.over_threshold_2
        ):
            self.over_threshold_2 = True
            print(f"Passed threshold 2 after {self.n_calls * self.n_procs}")
        elif (
            self.n_calls > self.steps_for_threshold_2
            and self.last_mean_reward < self.reward_threshold_2
            and not self.over_threshold_2
        ):
            continue_training = False
            print(
                f"Stop training because reward {self.last_mean_reward} is below {self.reward_threshold_2} after {self.n_calls * self.n_procs} steps"
            )
        return continue_training

    
    def selfplay_evaluation(self, ):
        if len(self.selfplay_callback.opponents[1:]) < 1: # if there are no copies of one self, skip
            return True
        new_opponents = self.selfplay_callback.opponents[1:][-self.eval_env.num_envs:] # take the last n selfs for evaluation
        if len(new_opponents) < self.eval_env.num_envs:
            new_opponents += [self.selfplay_callback.opponents[-1] for i in range(self.eval_env.num_envs - len(new_opponents))] # fill with the last copy
        self.eval_env.env_method("set_opponent", new_opponents) # set opponents to envs
        print("starting selfplay evaluation")
        # run evaluation
        episode_rewards, episode_lengths = evaluate_policy(   ### takes so long
            self.model,
            self.eval_env,
            n_eval_episodes=int(self.n_eval_episodes / 2),  # eval half the episodes
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
        )
        # set basic opponents again to envs
        self.eval_env.env_method("set_opponent", lh.BasicOpponent())

        mean_reward, std_reward = np.mean(episode_rewards), np.std(
            episode_rewards
        )
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
            episode_lengths
        )
        # self.last_mean_reward = mean_reward
        self.selfplay_last_mean_reward = mean_reward

        if self.verbose >= 1:
            print(
                f"SELFPLAY: Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(
                f"SELFPLAY: Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
            )
        # Add to current Logger
        self.logger.record("eval/selfplay_mean_reward", float(mean_reward))
        self.logger.record("eval/selfplay_mean_ep_length", mean_ep_length)

        if mean_reward > self.selfplay_best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(
                    os.path.join(
                        self.best_model_save_path, "selfplay_best_model"
                    )
                )
            # self.best_mean_reward = mean_reward
            self.selfplay_best_mean_reward = mean_reward
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                # save best selfplay model
                path = self.callback_on_new_best.path
                self.callback_on_new_best.path = path.replace(
                    "best_model", "selfplay_best_model"
                )
                self.callback_on_new_best.on_step()
                self.callback_on_new_best.path = path