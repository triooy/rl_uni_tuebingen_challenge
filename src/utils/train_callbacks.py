import os
import random

import gymnasium
import laserhockey.hockey_env as lh
import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from src.utils.wrapper import CustomWrapper
import logging

logger = logging.getLogger(__name__)


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
        env_mode=CustomWrapper.NORMAL,
        how_many_to_add=None,  # how many copies of a particular opponent to add to the list e.g.: 128 env add 2 copies of every past model
        # one model every 1 million steps, training for 20 million steps = 40 models
        # None = random
    ):
        super(SelfplayCallback, self).__init__(verbose)
        self.change_every_n_steps = change_every_n_steps
        self.add_opponent_every_n_steps = add_opponent_every_n_steps
        self.save_path = save_path
        self.n_envs = n_envs
        self.first_opponent_after_n_steps = first_opponent_after_n_steps
        self.how_many_to_add = how_many_to_add
        self.env_mode = env_mode
        self.opponents = []
        if self.env_mode == CustomWrapper.RANDOM:
            self.opponents.append(None)

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
                model_name = type(self.model).__name__
                # create empty file with model name
                with open(
                    os.path.join(f"{self.save_path}", f"{model_name}.txt"), "w"
                ) as f:
                    pass
                vec_env = self.model.get_vec_normalize_env()
                # if there is a normalization env, save that to
                if vec_env:
                    vec_env.save(
                        os.path.join(self.save_path, f"model_{self.n_calls}.pkl")
                    )
                self.opponents.append(path)
                if self.verbose > 0:
                    logger.info(f"Added self as opponent after {self.n_calls} steps")
            if (
                self.change_every_n_steps > 0
                and self.n_calls % self.change_every_n_steps == 0
            ):
                if self.how_many_to_add:  # how many copies to add
                    new_opponents = [
                        "lh.BasicOpponent(weak=False)",
                        "lh.BasicOpponent(weak=True)",
                    ]
                    for opponent in self.opponents:
                        new_opponents += [opponent for i in range(self.how_many_to_add)]
                    random.shuffle(new_opponents)
                    new_opponents = sorted(new_opponents, reverse=False)
                    for i in range(self.n_envs - len(new_opponents)):
                        new_opponents.append(lh.BasicOpponent(weak=False))
                    # replace 'lh.BasicOpponent(weak=False)' with lh.BasicOpponent(weak=False)
                    # and lh.BasicOpponent(weak=True) with lh.BasicOpponent(weak=True)
                    tmp = []
                    for opponent in new_opponents:
                        if opponent == "lh.BasicOpponent(weak=False)":
                            tmp.append(lh.BasicOpponent(weak=False))
                        elif opponent == "lh.BasicOpponent(weak=True)":
                            tmp.append(lh.BasicOpponent(weak=True))
                        else:
                            tmp.append(opponent)
                    new_opponents = tmp
                else:  # otherwise choose random
                    new_opponents = random.choices(
                        self.opponents,
                        k=self.n_envs,
                    )
                    new_opponents = [str(opponent) for opponent in new_opponents]
                    new_opponents = sorted(new_opponents, reverse=False)
                    new_opponents = [
                        opponent if opponent != "None" else None
                        for opponent in new_opponents
                    ]

                # set the opponents in the envs
                self.model.get_env().env_method("set_opponent", new_opponents)
                # log some stats
                count_basic_opponent = [
                    isinstance(opponent, (type(lh.BasicOpponent()), type(None)))
                    for opponent in new_opponents
                ].count(True)
                self.logger.record(
                    "eval/number_of_selfs", self.n_envs - count_basic_opponent
                )
                if self.verbose > 0:
                    logger.info(f"Changed opponent for envs after {self.n_calls} steps")
                    logger.info(
                        f"Number of basic opponents: {count_basic_opponent}, number of opponents: {len(self.opponents)}"
                    )
        return True

    def add_opponents(self, opponent):
        if isinstance(opponent, str):
            self.opponents.append(opponent)
        elif isinstance(opponent, list):
            self.opponents += opponent
        logger.info(f"Added opponent {opponent} to list of opponents")


class SaveEnv(BaseCallback):
    """
    Saves the environment if there is a new best model.
    """

    def __init__(self, verbose=0, path=None):
        super(SaveEnv, self).__init__(verbose)
        self.path = path

    def _on_step(self) -> bool:
        env = self.model.get_vec_normalize_env()
        # get base dir of save path
        base_dir = os.path.dirname(self.path)
        with open(f"{base_dir}/{type(self.model).__name__}.txt", "w") as f:
            pass
        if env:
            env.save(self.path)
            if self.verbose > 0:
                logger.info(f"Saved env to {self.path}")
        else:
            logger.info("No VecNormalize env to save")
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
        best_agents_env=None,
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
        self.selfplay_best_mean_reward = 0
        self.selfplay_last_mean_reward = 0
        self.best_agents_env = best_agents_env
        self.win_rate = 0
        self.draw_rate = 0
        self.loss_rate = 0

    def _super_on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )
            unique, counts = np.unique(episode_rewards, return_counts=True)
            wins, draws, losses = 0, 0, 0
            win = np.where(unique == 10)[0]
            draw = np.where(unique == 0)[0]
            loss = np.where(unique == -10)[0]
            if len(win) > 0:
                win = counts[win][0]
                wins = win / len(episode_rewards)
            if len(draw) > 0:
                draw = counts[draw][0]
                draws = draw / len(episode_rewards)
            if len(loss) > 0:
                loss = counts[loss][0]
                losses = loss / len(episode_rewards)
            self.logger.record("eval/wins", float(wins))
            self.logger.record("eval/draws", float(draws))
            self.logger.record("eval/losses", float(losses))
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = mean_reward
                self.win_rate = wins
                self.draw_rate = draws
                self.loss_rate = losses
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def _on_step(self) -> bool:
        continue_training = True
        # Do evaluation
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            continue_training = self._super_on_step()  # run normal evaluation

            ### Selfplay
            if self.best_agents_env:
                self.selfplay_evaluation()

            self.eval_idx += 1
            if self.trial:
                self.trial.report(self.last_mean_reward, self.eval_idx)
                # Prune trial if need.
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
        # check if reward is above threshold
        elif (
            self.n_calls < self.steps_for_threshold
            and self.last_mean_reward > self.reward_threshold
            and not self.over_threshold
        ):
            self.over_threshold = True
            logger.info(f"Passed threshold after {self.n_calls * self.n_procs}")
        # check if reward is below threshold
        elif (
            self.n_calls > self.steps_for_threshold
            and self.last_mean_reward < self.reward_threshold
            and not self.over_threshold
        ):
            continue_training = False
            logger.info(
                f"Stop training because reward {self.last_mean_reward} is below {self.reward_threshold} after {self.n_calls * self.n_procs} steps"
            )
        # check if reward is above threshold 2
        elif (
            self.n_calls < self.steps_for_threshold_2
            and self.last_mean_reward > self.reward_threshold_2
            and not self.over_threshold_2
        ):
            self.over_threshold_2 = True
            logger.info(f"Passed threshold 2 after {self.n_calls * self.n_procs}")
        # check if reward is below threshold 2
        elif (
            self.n_calls > self.steps_for_threshold_2
            and self.last_mean_reward < self.reward_threshold_2
            and not self.over_threshold_2
        ):
            continue_training = False
            logger.info(
                f"Stop training because reward {self.last_mean_reward} is below {self.reward_threshold_2} after {self.n_calls * self.n_procs} steps"
            )
        return continue_training

    def selfplay_evaluation(
        self,
    ):
        logger.info("starting selfplay evaluation")
        # run evaluation

        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.best_agents_env)
            except AttributeError:
                logger.info("Could not sync envs")

        episode_rewards, episode_lengths = evaluate_policy(  ### takes so long
            self.model,
            self.best_agents_env,
            n_eval_episodes=self.n_eval_episodes,  # eval half the episodes
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
            episode_lengths
        )
        # self.last_mean_reward = mean_reward
        self.selfplay_last_mean_reward = mean_reward

        if self.verbose >= 1:
            logger.info(
                f"SELFPLAY: Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            logger.info(
                f"SELFPLAY: Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
            )
        # Add to current Logger
        self.logger.record("eval/selfplay_mean_reward", float(mean_reward))
        self.logger.record("eval/selfplay_mean_ep_length", mean_ep_length)

        if mean_reward > self.selfplay_best_mean_reward:
            if self.verbose >= 1:
                logger.info("New best mean reward!")
            if self.best_model_save_path is not None:
                model_name = type(self.model).__name__
                with open(
                    os.path.join(f"{self.best_model_save_path}", f"{model_name}.txt"),
                    "w",
                ) as f:
                    pass
                self.model.save(
                    os.path.join(self.best_model_save_path, "selfplay_best_model")
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


class StopTrainingOnNoModelImprovementWithSelfplay(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward for basic opponent and selfplay) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """

    def __init__(
        self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.max_no_improvement_evals_selfplay = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.last_best_mean_reward_selfplay = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert (
            self.parent is not None
        ), "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True

        if self.n_calls > self.min_evals:
            if (
                self.parent.best_mean_reward > self.last_best_mean_reward
                and self.parent.selfplay_best_mean_reward
                > self.last_best_mean_reward_selfplay
            ):
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward
        self.last_best_mean_reward_selfplay = self.parent.selfplay_best_mean_reward

        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )

        return continue_training
