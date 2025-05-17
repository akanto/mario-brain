import gymnasium as gym
import numpy as np

from mario_brain.reward_summary import CumulativeRewardWrapper


class MockEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(4,), dtype=np.uint8
        )
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return np.array([0, 0, 0, 0]), {}

    def step(self, action):
        self.current_step += 1
        reward = 1.0
        terminated = self.current_step >= 5
        truncated = False
        info = {"step": self.current_step}
        return np.array([0, 0, 0, 0]), reward, terminated, truncated, info


def test_cumulative_reward_wrapper():
    env = MockEnv()
    wrapped_env = CumulativeRewardWrapper(env)

    obs, info = wrapped_env.reset()
    assert wrapped_env.cumulative_reward == 0
    assert wrapped_env.episode_count == 1

    total_reward = 0
    for _ in range(5):
        obs, reward, terminated, truncated, info = wrapped_env.step(0)
        total_reward += reward
        assert wrapped_env.cumulative_reward == total_reward

    assert terminated
    assert wrapped_env.cumulative_reward == 5.0

    obs, info = wrapped_env.reset()
    assert wrapped_env.cumulative_reward == 0
    assert wrapped_env.episode_count == 2
