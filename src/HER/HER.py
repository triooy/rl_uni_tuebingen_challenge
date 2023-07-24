import copy
import numpy as np
import torch as tf
from gymnasium import spaces
import math

# we need this to not break stable baseline implementation of off policy algorithms with replay buffers
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from src.HER.utils import get_action_dim, get_obs_shape

""" Custom HER Replay Buffer. Some parts are influenced by stable baseline implementation of DictBufferClass in order
    to not break stable baseline implementation of off policy algorithms with replay buffers. """


class CustomHindisghtExperienceReplay:
    """
    :param buffer_size: Buffer size
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: Training environment
    :param device: Devcie for Pytorch
    :param n_envs: Number of parallel environments
    :param her_ratio: Ratio between HER replays and regular replays (between 0 and 1)
    :param optimize_memory_usage: this param is only needed to not break stable baseline implementation of off policy algorithms
    """

    def __init__(
        self,
        buffer_size,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env,
        device="cpu",
        n_envs=1,
        her_ratio=0.75,
        action_space_dimension=4,
        observation_length=18,
        optimize_memory_usage=False,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        # self.obs_shape = get_obs_shape(observation_space)
        self.obs_shape = {
            "achieved_goal": (observation_length,),
            "desired_goal": (observation_length,),
            "observation": (observation_length,),
        }
        # Make sure we are in dict observation space
        assert isinstance(
            self.obs_shape, dict
        ), "DictReplayBuffer must be used with Dict obs space only"

        self.action_space_dimension = action_space_dimension
        self.position = 0
        self.buffer_full = False
        self.device = device
        self.n_envs = n_envs
        self.observation_length = observation_length
        self.buffer_size = buffer_size // n_envs

        """ Initialization of observations, future"""

        self.observations = {
            key: np.zeros(
                (self.buffer_size, self.n_envs, self.observation_length),
                dtype=np.float32,
            )
            for key in self.obs_shape.keys()
        }
        self.future_observation = {
            key: np.zeros(
                (self.buffer_size, self.n_envs, self.observation_length),
                dtype=np.float32,
            )
            for key in self.obs_shape.keys()
        }
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_space_dimension),
            dtype=action_space.dtype,
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.infos = np.array(
            [[{} for i in range(self.n_envs)] for j in range(self.buffer_size)]
        )
        self.env = env
        self.her_ratio = her_ratio

        # save episode information in order to create correct virtual samples later
        self.episode_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.current_episode_start = np.zeros(self.n_envs, dtype=np.int64)

    def to_torch(self, array):
        return tf.tensor(array, device=self.device)

    def get_episode_length(self, env_idx):
        """
        Compute and store the episode length for environment with index env_idx
        """
        episode_start = self.current_episode_start[env_idx]
        episode_end = self.position
        if episode_end < episode_start:
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        self.current_episode_start[env_idx] = self.position
        self.episode_length[episode_indices, env_idx] = episode_end - episode_start

    def get_observations_and_next_observations(self, batch_indices, env_indices):

        observations = self.env.normalize_obs(
            {
                tmp_key: tmp_obs[batch_indices, env_indices, :]
                for tmp_key, tmp_obs in self.observations.items()
            }
        )
        next_observations = self.env.normalize_obs(
            {
                tmp_key: tmp_obs[batch_indices, env_indices, :]
                for tmp_key, tmp_obs in self.future_observation.items()
            }
        )
        return observations, next_observations

    def sample(self, batch_size, env=None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        episode_long_enough = np.where(self.episode_length > 0, True, False)

        episode_indices = np.flatnonzero(episode_long_enough)
        # sample for batch size
        sampled_indices = np.random.choice(
            episode_indices, size=batch_size, replace=True
        )
        batch_indices, environment_indices = np.unravel_index(
            sampled_indices, episode_long_enough.shape
        )

        # Split between normal real samples and ones that are created additionally
        number_of_additional_samples = [int(self.her_ratio * batch_size)]
        additional_batch_indices, real_batch_indices = np.split(
            batch_indices, number_of_additional_samples
        )
        additional_environment_indices, real_environment_indices = np.split(
            environment_indices, number_of_additional_samples
        )

        # Generate real samples
        real_data = self.generate_real_samples(
            real_batch_indices, real_environment_indices, env
        )
        # Generate additional samples
        generated_samples = self.generate_additional_samples(
            additional_batch_indices, additional_environment_indices, env
        )

        observations = {
            key: tf.cat(
                (real_data.observations[key], generated_samples.observations[key])
            )
            for key in generated_samples.observations.keys()
        }
        actions = tf.cat((real_data.actions, generated_samples.actions))
        next_observations = {
            key: tf.cat(
                (
                    real_data.next_observations[key],
                    generated_samples.next_observations[key],
                )
            )
            for key in generated_samples.next_observations.keys()
        }
        dones = tf.cat((real_data.dones, generated_samples.dones))
        rewards = tf.cat((real_data.rewards, generated_samples.rewards))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def generate_real_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env=None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        observations, next_observations = self.get_observations_and_next_observations(
            batch_indices, env_indices
        )

        # Convert to torch tensor
        observations = {
            tmp_key: self.to_torch(tmp_obs) for tmp_key, tmp_obs in observations.items()
        }
        next_observations = {
            tmp_key: self.to_torch(tmp_obs)
            for tmp_key, tmp_obs in next_observations.items()
        }

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(self.dones[batch_indices, env_indices]).reshape(-1, 1),
            rewards=self.to_torch(
                self.env.normalize_reward(
                    self.rewards[batch_indices, env_indices].reshape(-1, 1)
                ).astype(np.float32)
            ),
        )

    def generate_additional_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env=None,
    ) -> DictReplayBufferSamples:
        """
        Generate additional samples for the given batch and environment indices.
        """
        # we need infos, observations and next_observations to compute the new rewards
        infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        observations, next_observations = self.get_observations_and_next_observations(
            batch_indices, env_indices
        )

        # Sample and set new goals
        new_goals = self.select_goals(batch_indices, env_indices)
        observations["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_observations["desired_goal"] = new_goals

        """
        Call the compute reward function from the environment.
        We need an vectorized implementation of the compute reward function that takes three input arguments.
        The corresponding function is used in the wrapper implementation.
        The implementation of the compute reward function itself canbe found in the HER/utils.py file.
        """
        rewards = self.env.env_method(
            "compute_reward",
            next_observations["achieved_goal"],
            observations["desired_goal"],
            infos,
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)
        observations = self.env.normalize_obs(observations)
        next_observations = self.env.normalize_obs(next_observations)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in observations.items()}
        next_observations = {
            key: self.to_torch(obs) for key, obs in next_observations.items()
        }

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(self.dones[batch_indices, env_indices]).reshape(-1, 1),
            rewards=self.to_torch(
                self.env.normalize_reward(rewards.reshape(-1, 1)).astype(np.float32)
            ),
        )

    def select_goals(self, batch_indices, env_indices):
        """
        Select the goals for the generated samples.
        """
        batch_episode_start = self.episode_start[batch_indices, env_indices]
        batch_episode_length = self.episode_length[batch_indices, env_indices]

        indices = batch_indices - batch_episode_start
        # check if the indices are in the valid range of buffer indices
        indices = indices % self.buffer_size

        episode_indices = np.random.randint(indices, batch_episode_length)
        new_goal_indices = (episode_indices + batch_episode_start) % self.buffer_size

        # Retrieve the "achieved_goal" data from self.future_observation
        # corresponding to the computed transition_indices and env_indices
        selected_goals = self.future_observation["achieved_goal"][
            new_goal_indices, env_indices
        ]
        return selected_goals

    """ Helper functions imitated closely from Stablebaselines3
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
    """

    def add_helper(
        self,
        observations,
        next_observations,
        action,
        reward,
        done,
        infos,
    ):
        """Imitates add function from stable baseline implementation of DictBufferClass"""
        for key in self.observations.keys():
            observations[key] = observations[key].reshape(
                (self.n_envs,) + self.obs_shape[key]
            )

        for key in self.future_observation.keys():
            self.future_observation[key][self.position] = np.array(
                next_observations[key]
            ).copy()

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_space_dimension))

        self.actions[self.position] = np.array(action).copy()
        self.rewards[self.position] = np.array(reward).copy()
        self.dones[self.position] = np.array(done).copy()

        self.position = self.position + 1
        if self.position == self.buffer_size:
            self.buffer_full = True
            self.position = 0

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos,
    ):
        """Imitates add function from stable baseline implementation of DictBufferClass"""
        for env_idx in range(self.n_envs):
            episode_start = self.episode_start[self.position, env_idx]
            episode_length = self.episode_length[self.position, env_idx]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = (
                    np.arange(self.position, episode_end) % self.buffer_size
                )
                self.episode_length[episode_indices, env_idx] = 0

        # Update episode start
        self.episode_start[self.position] = self.current_episode_start.copy()

        self.infos[self.position] = infos
        self.add_helper(obs, next_obs, action, reward, done, infos)

        # Episode ending
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self.get_episode_length(env_idx)
