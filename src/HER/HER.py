import copy
import numpy as np
import torch as th
from gymnasium import spaces

# we need this to not break stable baseline implementation of off policy algorithms with replay buffers
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from src.HER.utils import get_action_dim, get_obs_shape


class CustomHerReplayBuffer:
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
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env,
        device="cpu",
        n_envs=1,
        her_ratio=0.75,
        optimize_memory_usage=False,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        # Make sure we are in dict observation space
        assert isinstance(
            self.obs_shape, dict
        ), "DictReplayBuffer must be used with Dict obs space only"

        self.action_dim = get_action_dim(action_space)
        self.position = 0
        self.buffer_full = False
        self.device = device
        self.n_envs = n_envs

        self.buffer_size = buffer_size // n_envs
        self.buffer_size = max(self.buffer_size, 1)

        self.observations = {
            key: np.zeros(
                (self.buffer_size, self.n_envs, *_obs_shape),
                dtype=observation_space[key].dtype,
            )
            for key, _obs_shape in self.obs_shape.items()
        }
        self.future_observation = {
            key: np.zeros(
                (self.buffer_size, self.n_envs, *_obs_shape),
                dtype=observation_space[key].dtype,
            )
            for key, _obs_shape in self.obs_shape.items()
        }
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        # self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = np.array(
            [[{} for i in range(self.n_envs)] for j in range(self.buffer_size)]
        )
        # infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        self.env = env
        self.her_ratio = her_ratio

        # save episode information in order to create correct virtual samples later
        self.episode_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.current_episode_start = np.zeros(self.n_envs, dtype=np.int64)

    def to_torch(self, arr):
        return th.tensor(arr, device=self.device)

    def add_helper(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        infos,
    ):
        for key in self.observations.keys():
            # Reshape for parallel environments
            obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])

        for key in self.future_observation.keys():
            self.future_observation[key][self.position] = np.array(next_obs[key]).copy()

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.position] = np.array(action).copy()
        self.rewards[self.position] = np.array(reward).copy()
        self.dones[self.position] = np.array(done).copy()

        self.position += 1
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
        # When the buffer is full, we rewrite on old episodes. When we start to
        # rewrite on an old episodes, we want the whole old episode to be deleted
        # (and not only the transition on which we rewrite). To do this, we set
        # the length of the old episode to 0, so it can't be sampled anymore.
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
        # Store the transition
        # super().add(obs, next_obs, action, reward, done, infos)
        # NEW
        self.add_helper(obs, next_obs, action, reward, done, infos)

        # When episode ends, compute and store the episode length
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                self.get_episode_length(env_idx)

    def get_episode_length(self, env_idx):
        """
        Compute and store the episode length for environment with index env_idx
        """
        episode_start = self.current_episode_start[env_idx]
        episode_end = self.position
        if episode_end < episode_start:
            # Reset Buffer when full
            episode_end += self.buffer_size
        episode_indices = np.arange(episode_start, episode_end) % self.buffer_size
        self.current_episode_start[env_idx] = self.position
        self.episode_length[episode_indices, env_idx] = episode_end - episode_start

    def sample(self, batch_size: int, env=None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: Associated VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.
        is_valid = self.episode_length > 0
        if not np.any(is_valid):
            raise RuntimeError(
                "Unable to sample before the end of the first episode. We recommend choosing a value "
                "for learning_starts that is greater than the maximum number of timesteps in the environment."
            )
        # Get the indices of valid transitions
        # Example:
        # if is_valid = [[True, False, False], [True, False, True]],
        # is_valid has shape (buffer_size=2, n_envs=3)
        # then valid_indices = [0, 3, 5]
        # they correspond to is_valid[0, 0], is_valid[1, 0] and is_valid[1, 2]
        # or in numpy format ([rows], [columns]): (array([0, 1, 1]), array([0, 0, 2]))
        # Those indices are obtained back using np.unravel_index(valid_indices, is_valid.shape)
        valid_indices = np.flatnonzero(is_valid)
        # Sample valid transitions that will constitute the minibatch of size batch_size
        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=True)
        # Unravel the indexes, i.e. recover the batch and env indices.
        # Example: if sampled_indices = [0, 3, 5], then batch_indices = [0, 1, 1] and env_indices = [0, 0, 2]
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

        # Split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_indices, real_batch_indices = np.split(
            batch_indices, [nb_virtual]
        )
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # Get real and virtual data
        real_data = self._get_real_samples(real_batch_indices, real_env_indices, env)
        # Create virtual transitions by sampling new desired goals and computing new rewards
        virtual_data = self._get_virtual_samples(
            virtual_batch_indices, virtual_env_indices, env
        )

        # Concatenate real and virtual data
        observations = {
            key: th.cat((real_data.observations[key], virtual_data.observations[key]))
            for key in virtual_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions))
        next_observations = {
            key: th.cat(
                (real_data.next_observations[key], virtual_data.next_observations[key])
            )
            for key in virtual_data.next_observations.keys()
        }
        dones = th.cat((real_data.dones, virtual_data.dones))
        rewards = th.cat((real_data.rewards, virtual_data.rewards))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def _get_real_samples(
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
        obs_ = self.env.normalize_obs(
            {
                key: obs[batch_indices, env_indices, :]
                for key, obs in self.observations.items()
            }
        )
        next_obs_ = self.env.normalize_obs(
            {
                key: obs[batch_indices, env_indices, :]
                for key, obs in self.future_observation.items()
            }
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices]
                * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self.env.normalize_reward(
                    self.rewards[batch_indices, env_indices].reshape(-1, 1)
                ).astype(np.float32)
            ),
        )

    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env=None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {
            key: obs[batch_indices, env_indices, :]
            for key, obs in self.observations.items()
        }
        next_obs = {
            key: obs[batch_indices, env_indices, :]
            for key, obs in self.future_observation.items()
        }

        # The copy may cause a slow down
        infos = copy.deepcopy(self.infos[batch_indices, env_indices])
        # infos = [{} for _ in range(len(batch_indices))]

        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        # Compute new reward
        # Use vectorized compute reward function
        """
        rewards = self.compute_reward(
            next_obs["achieved_goal"], obs["desired_goal"], infos
        )
        """
        rewards = self.env.env_method(
            "compute_reward",
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
            # we use the method of the first environment assuming that all environments are identical.
            indices=[0],
        )
        rewards = rewards[0].astype(np.float32)
        obs = self.env.normalize_obs(obs)
        next_obs = self.env.normalize_obs(next_obs)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_indices, env_indices]
                * (1 - self.timeouts[batch_indices, env_indices])
            ).reshape(-1, 1),
            rewards=self.to_torch(
                self.env.normalize_reward(rewards.reshape(-1, 1)).astype(np.float32)
            ),
        )

    def _sample_goals(
        self, batch_indices: np.ndarray, env_indices: np.ndarray
    ) -> np.ndarray:
        """

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :return: Sampled goals
        """
        batch_episode_start = self.episode_start[batch_indices, env_indices]
        batch_episode_length = self.episode_length[batch_indices, env_indices]

        # select sample with future goal selection strate
        current_indices_in_episode = (
            batch_indices - batch_episode_start
        ) % self.buffer_size
        transition_indices_in_episode = np.random.randint(
            current_indices_in_episode, batch_episode_length
        )
        transition_indices = (
            transition_indices_in_episode + batch_episode_start
        ) % self.buffer_size
        return self.future_observation["achieved_goal"][transition_indices, env_indices]
