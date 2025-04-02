import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.n_envs = None
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.current_rewards = np.zeros(self.n_envs)
        self.current_lengths = np.zeros(self.n_envs)

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        self.current_rewards += rewards
        self.current_lengths += 1

        for i in range(self.n_envs):
            if dones[i]:
                # Log reward and length for each env
                ep_reward = self.current_rewards[i]
                ep_length = self.current_lengths[i]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Log to TensorBoard
                self.logger.record("episode/cumulative_reward", ep_reward)
                self.logger.record("episode/length", ep_length)

                # Reset that env's counters
                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0

        return True
