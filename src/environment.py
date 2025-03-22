import gym_super_mario_bros
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from gym.wrappers import GrayScaleObservation

from nes_py.wrappers import JoypadSpace

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import gym

def _register_custom_mario_env(id, **kwargs):
    """
    Register a Super Mario Bros  environment with OpenAI Gym.

    Args:
        id (str): id for the env to register
        kwargs (dict): keyword arguments for the SuperMarioBrosEnv initializer

    Returns:
        None

    """
    # register the environment
    gym.envs.registration.register(
        id=id,
        entry_point='environment:CustomMarioEnv',
        max_episode_steps=9999999,
        reward_threshold=9999999,
        kwargs=kwargs,
        nondeterministic=True,
    )

_register_custom_mario_env('CustomSuperMarioBros-v0', rom_mode='vanilla')
_register_custom_mario_env('CustomSuperMarioBros-v1', rom_mode='downsample')
_register_custom_mario_env('CustomSuperMarioBros-v2', rom_mode='pixel')
_register_custom_mario_env('CustomSuperMarioBros-v3', rom_mode='rectangle')

class CustomMarioEnv(SuperMarioBrosEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action = None
        self._last_action_repetition_count = 0

    @property
    def _time_penalty(self):
        """Return the reward for the in-game clock ticking."""
        self._time_last = self._time
        # Overwrite the time penalty with a small negative number for each step
        _reward = -0.05
        return _reward
    
    @property
    def _repetitive_action_penalty(self):
        """Penalty for repeating the same action without moving."""
        # Apply a penalty only if stuck and repeating an action too long
        if self._last_action_repetition_count > 5 and self._x_reward <= 0:
            # print(f"Repetitive action penalty: -1 (count: {self._last_action_repetition_count})")
            return -1  # Penalize for repetitive non-progressive actions
        return 0
    
    def _register_action(self, action):
        if self._last_action == action:
            self._last_action_repetition_count += 1
        else:
            self._last_action_repetition_count = 0
        self._last_action = action
    
    def step(self, action):
        """Override the step method to capture the action taken."""
        self._register_action(action)
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _get_reward(self):
        """Return the reward after a step occurs."""
        _reward = super()._get_reward() + self._repetitive_action_penalty
        # print(f"Total reward: {_reward}")
        return _reward

def create_env():
    # env = gym.make('CustomSuperMarioBros-v0')
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def create_training_env():
    env = create_env()
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env
