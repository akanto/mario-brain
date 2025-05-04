import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class CumulativeRewardWrapper(gym.RewardWrapper):
    #def __init__(self, env, writer: SummaryWriter):
    def __init__(self, env):
        super().__init__(env)
        # self.writer = writer
        self.episode_count = 0
        self.cumulative_reward = 0
        self.last_info = {}

    def reset(self, **kwargs):
        # Log the cumulative reward to TensorBoard after the episode ends
        if self.episode_count > 0:
            # self.writer.add_scalar("Cumulative Reward", self.cumulative_reward, self.episode_count)
            info_str = ", ".join(f"{key}: {value}" for key, value in self.last_info.items() if not isinstance(value, np.ndarray))
            print(f"Episode {self.episode_count}, Cumulative Reward: {self.cumulative_reward}, Info: {info_str}")
        self.episode_count += 1
        self.cumulative_reward = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        # Accumulate the reward here
        self.cumulative_reward += reward
        # print(f"Reward: {reward}, Cumulative Reward: {self.cumulative_reward}")
        return reward  # Return the original or modified reward
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.last_info = info
        return observation, reward, terminated, truncated, info
