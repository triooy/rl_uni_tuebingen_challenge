import os
from typing import Union
import time
import pandas as pd
import shutil

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.ppo.policies import MlpPolicy as PPO_MlpPolicy
from stable_baselines3.td3.policies import MlpPolicy as TD3_MlpPolicy

from src.hyperparameter.hyperparams import *
from src.utils.train_callbacks import SaveEnv, SelfplayCallback, TrialEvalCallback
from src.utils.wrapper import CustomWrapper, get_env
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        agent: Union[PPO, TD3],
        policy: str,
        n_timesteps: int = 1_000_000,
        negative_reward: bool = True,
        discrete_action_space: bool = False,
        normalize: bool = True,
        n_train_envs: int = 128,
        n_eval_envs: int = 10,
        best_agents_path: str = None,
        agents_kwargs: Dict[str, Any] = None,
        max_no_improvement_evals: int = 6,
        min_evals: int = 6,
        n_eval_episodes: int = 500,
        n_evaluations: int = 6,
        reward_threshold: float = -np.inf,
        reward_threshold_2: float = -np.inf,
        steps_for_threshold: int = 0,
        steps_for_threshold_2: int = 0,
        selfplay: bool = False,
        add_opponent_every_n_steps: int = 0,
        change_opponent_every_n_steps: int = 128000,
        first_opponent_after_n_steps: int = 1_000_000,
        how_many_to_add: int = 1,
        tensorboard_log_dir: str = None,
        run_name: str = None,
        trial: optuna.Trial = None,
        csv_filename: str = None,
        verbose: int = 1,
        add_to_best_agents_when_mean_reward_is_above=None,
        add_to_best_agents_when_best_agents_mean_reward_is_above=None,
        start_method: str = "fork",
        **kwargs,
    ) -> None:
        self.negative_reward = negative_reward
        self.discrete_action_space = discrete_action_space
        self.n_train_envs = n_train_envs
        self.n_eval_envs = n_eval_envs
        self.normalize = normalize
        self.verbose = verbose
        self.tensorboard_log_dir = tensorboard_log_dir
        self.trial = trial
        self.run_name = run_name
        self.save_path = os.path.join(tensorboard_log_dir, run_name)
        self.csv_filename = csv_filename
        self.start_method = start_method

        # create environments
        self.create_environments()

        # training
        self.n_timesteps = n_timesteps

        # evaluation parameters
        self.n_eval_episodes = n_eval_episodes
        self.n_evaluations = n_evaluations
        self.eval_freq = int(
            self.n_timesteps / (self.n_evaluations * self.n_train_envs)
        )
        self.reward_threshold = reward_threshold
        self.reward_threshold_2 = reward_threshold_2
        self.steps_for_threshold = int(steps_for_threshold / self.n_train_envs)
        self.steps_for_threshold_2 = int(steps_for_threshold_2 / self.n_train_envs)

        # selfplay
        self.selfplay = selfplay
        self.add_opponent_every_n_steps = int(
            add_opponent_every_n_steps / self.n_train_envs
        )
        self.change_opponent_every_n_steps = int(
            change_opponent_every_n_steps / self.n_train_envs
        )
        self.first_opponent_after_n_steps = int(
            first_opponent_after_n_steps / self.n_train_envs
        )
        self.how_many_to_add = how_many_to_add

        # best agents
        self.best_agents = []
        self.best_agents_path = best_agents_path
        self.best_agents_env = None
        if self.best_agents_path:
            self.load_best_agents()

        # normalize env
        if self.normalize:
            self.normalize_env()

        # agent parameters
        self.policies = {
            "PPO_MlpPolicy": PPO_MlpPolicy,
            "TD3_MlpPolicy": TD3_MlpPolicy,
        }
        self.agent = agent
        self.model_type = agent
        self.policy = policy
        self.agents_kwargs = agents_kwargs
        self.setup_agent()

        # callbacks
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.callbacks = []
        self.stop_train_callback = None
        self.save_env_callback = None
        self.selfplay_callback = None
        self.setup_callbacks()

        # training results
        self.mean_reward = -np.inf
        self.std_reward = 0
        self.best_agent_mean_reward = -np.inf
        self.best_agent_std_reward = 0
        self.train_time = 0
        self.add_to_best_agents_when_best_agents_mean_reward_is_above = (
            add_to_best_agents_when_best_agents_mean_reward_is_above
        )
        self.add_to_best_agents_when_mean_reward_is_above = (
            add_to_best_agents_when_mean_reward_is_above
        )

    def setup_agent(self):
        """Setup agent"""
        logger.info("Setting up agent...")
        if self.agent == "PPO":
            self.policy = self.policies[self.policy]
            self.agents_kwargs["policy"] = self.policy
            self.agents_kwargs["env"] = self.train_env
            self.agents_kwargs["verbose"] = self.verbose
            self.agents_kwargs["tensorboard_log"] = self.tensorboard_log_dir
            self.agent = PPO(**self.agents_kwargs)
            logger.info(f"PPO Agent parameters: {self.agents_kwargs}")
            logger.info("PPO Agent setup complete...")
        elif self.agent == "TD3":
            self.policy = self.policies[self.policy]
            self.agents_kwargs["policy"] = self.policy
            self.agents_kwargs["env"] = self.train_env
            self.agents_kwargs["verbose"] = self.verbose
            self.agents_kwargs["tensorboard_log"] = self.tensorboard_log_dir
            self.agent = TD3(**self.agents_kwargs)
            logger.info(f"TD3 Agent parameters: {self.agents_kwargs}")
            logger.info("TD3 Agent setup complete...")

    def setup_callbacks(self):
        """Setup callbacks for training"""
        # Stop training if there is no improvement after more than 5 evaluations
        logger.info("Setting up callbacks...")
        self.stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=self.max_no_improvement_evals,
            min_evals=self.min_evals,
            verbose=self.verbose,
        )
        if self.normalize:
            # Save the environment that was used for training
            # This is important because the running mean and std are not saved in the model
            self.save_env_callback = SaveEnv(
                path=os.path.join(self.save_path, "best_model_env.pkl")
            )
        # Create the callback that will periodically evaluate and report the performance.
        self.eval_callback = TrialEvalCallback(
            self.eval_env,  # evaluation environment
            self.trial,  # optuna trial (for reporting)
            best_model_save_path=self.save_path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=self.eval_freq,
            deterministic=True,
            callback_after_eval=self.stop_train_callback,
            callback_on_new_best=self.save_env_callback,
            verbose=self.verbose,
            reward_threshold=self.reward_threshold,
            reward_threshold_2=self.reward_threshold_2,
            steps_for_threshold_2=self.steps_for_threshold_2,
            steps_for_threshold=self.steps_for_threshold,
            best_agents_env=self.best_agents_env,
        )

        self.callbacks.append(self.eval_callback)
        if self.selfplay:
            # Create the callback that will periodically add a copy of the agent as opponent
            # and update the opponent every n steps
            self.selfplay_callback = SelfplayCallback(
                verbose=self.verbose,
                add_opponent_every_n_steps=self.add_opponent_every_n_steps,
                change_every_n_steps=self.change_opponent_every_n_steps,
                save_path=self.save_path,
                n_envs=self.n_train_envs,
                first_opponent_after_n_steps=self.first_opponent_after_n_steps,
                how_many_to_add=self.how_many_to_add,
            )
            if self.best_agents_path:
                if len(self.best_agents) == 0:
                    self.load_best_agents()

                self.selfplay_callback.add_opponents(self.best_agents)
            self.callbacks.append(
                self.selfplay_callback
            )  # add selfplay callback to training callbacks

    def train(self):
        """Train the agent"""
        # reset environments
        logger.info("Resetting environments...")
        self.train_env.reset()
        self.eval_env.reset()
        if self.best_agents_env:
            self.best_agents_env.reset()

        logger.info("Starting training...")
        start_time = time.time()
        self.agent.learn(
            total_timesteps=self.n_timesteps,
            callback=self.callbacks,
            tb_log_name=self.run_name,
        )
        self.train_time = time.time() - start_time
        # Evaluate the final model
        self.mean_reward = self.eval_callback.best_mean_reward
        logger.info(f"mean_reward:{self.mean_reward:.2f}")

        if self.best_agents_path:
            self.best_agent_mean_reward = self.eval_callback.selfplay_best_mean_reward
            logger.info(f"best_agents_mean_reward:{self.best_agent_mean_reward:.2f} ")

        self.write_csv(os.path.join(self.save_path, self.csv_filename))
        if (
            self.add_to_best_agents_when_best_agents_mean_reward_is_above
            and self.add_to_best_agents_when_best_agents_mean_reward_is_above
            < self.best_agent_mean_reward
            and self.add_to_best_agents_when_mean_reward_is_above < self.mean_reward
        ):
            self.copy_trained_agent_to_best_agents()
        self.eval_env.close()
        self.train_env.close()
        if self.best_agents_env:
            self.best_agents_env.close()

    def evaluate(self, n_eval_episodes: int = 500):
        """Evaluate the agent"""
        mean_reward, std_reward = evaluate_policy(
            self.agent,
            self.eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        logger.info(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        if self.best_agents_path:
            best_agent_mean_reward, best_agent_std_reward = evaluate_policy(
                self.agent,
                self.best_agents_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            logger.info(
                f"mean_reward:{best_agent_mean_reward:.2f} +/- {best_agent_std_reward:.2f}"
            )
        return mean_reward, std_reward

    def normalize_env(self):
        """Normalize the environments"""
        logger.info("Normalizing environments...")
        self.train_env = VecNormalize(
            self.train_env,
            norm_obs=True,
            norm_reward=True,
        )
        self.eval_env = VecNormalize(
            self.eval_env,
            norm_obs=True,
            norm_reward=False,  # reward is not normalized for evaluation
            training=False,  # training is set to false, so that the running mean and std are not updated
        )
        if self.best_agents_env:
            self.best_agents_env = VecNormalize(
                self.best_agents_env,
                norm_obs=True,
                norm_reward=False,  # reward is not normalized for evaluation
                training=False,  # training is set to false, so that the running mean and std are not updated
            )

    def load_best_agents(self):
        """Load best agents from best_agent_path"""
        # list all best agents
        logger.info("Loading best agents...")
        best_agents = os.listdir(self.best_agents_path)
        if len(best_agents) != 0:
            self.best_agents = [
                os.path.join(self.best_agents_path, agent, "best_model.zip")
                for agent in best_agents
                if os.path.isdir(os.path.join(self.best_agents_path, agent))
            ]
            logger.info(f"Loaded {len(self.best_agents)} best agents...")
            # create eval envs for best agents
            self.best_agents = get_env(
                n_envs=len(self.best_agents),
                mode=CustomWrapper.NORMAL,
                discrete_action_space=self.discrete_action_space,
                negativ_reward=False,
                weak=False,
                start_method=self.start_method,
            )
            # load best agents
            for i, agent in enumerate(self.best_agents):
                self.best_agents_env.env_method("set_opponent", agent, indices=[i])

    def create_environments(self):
        """Create train and eval environments"""
        logger.info("Creating environments...")
        self.train_env = get_env(
            self.n_train_envs,
            mode=None,
            discrete_action_space=self.discrete_action_space,
            negativ_reward=self.negative_reward,
            weak=False,
            start_method=self.start_method,
        )
        self.eval_env = get_env(
            self.n_eval_envs,
            mode=CustomWrapper.NORMAL,
            discrete_action_space=self.discrete_action_space,
            negative_reward=False,
            weak=False,
            start_method=self.start_method,
        )

    def write_csv(self, path):
        data = {
            "params": [str(self.agents_kwargs)],
            "normalize": [self.normalize],
            "negativ_reward": [self.negative_reward],
            "discrete_action_space": [self.discrete_action_space],
            "training_time": [self.train_time],
            "mean_reward": [self.mean_reward],
            "std_reward": [self.std_reward],
            "model_type": [self.model_type],
            "policy": [self.policy],
            "n_timesteps": [self.n_timesteps],
            "n_train_envs": [self.run_name],
        }
        if self.selfplay:
            data["best_agent_mean_reward"] = [self.best_agent_mean_reward]
            data["best_agent_std_reward"] = [self.best_agent_std_reward]
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        # check if file exists
        if os.path.isfile(path):
            df.to_csv(path, mode="a", header=False)
        else:
            df.to_csv(path, mode="a", header=True)

    def copy_trained_agent_to_best_agents(self):
        """Copy trained agent to best_agents folder"""
        # copy selfplay_best_model.zip, PPO.txt and selfplay_best_model_env.pkl to best_agents
        logger.info("Copying trained agent to best_agents folder...")
        if self.best_agents_path:
            save_path = os.path.join(self.best_agents_path, self.run_name)
        if os.path.isdir(save_path):
            save_path += "_1"
        os.mkdir(save_path)
        model_prefix = "selfplay_"
        # check if selfplay_best_model.zip exists
        selfplay_best_model = os.path.join(self.save_path, "selfplay_best_model.zip")
        if not os.path.isfile(selfplay_best_model):
            model_prefix = ""

        shutil.copy(
            os.path.join(self.save_path, f"{model_prefix}best_model.zip"),
            os.path.join(save_path, "best_model.zip"),
        )
        shutil.copy(
            os.path.join(self.save_path, f"{self.model_type}.txt"),
            os.path.join(save_path, f"{self.model_type}.txt"),
        )
        shutil.copy(
            os.path.join(self.save_path, self.csv_filename),
            os.path.join(save_path, self.csv_filename),
        )
        if self.normalize:
            shutil.copy(
                os.path.join(self.save_path, f"{model_prefix}best_model_env.pkl"),
                os.path.join(save_path, "best_model_env.pkl"),
            )
