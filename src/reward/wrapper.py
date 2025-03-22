# This file contains the custom reward wrapper for the environment.
import gym
from gym import Env
from typing import Callable

class RewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper for the environment.

    Args:
        env: the environment to wrap
        reward_fn: the custom reward function to use
    Returns:
        None
    """
    def __init__(self, env: Env, reward_fn: Callable[[float, dict], float]):
        super().__init__(env)
        self.reward_fn = reward_fn
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward_fn(reward, info)
        return obs, reward, done, info


