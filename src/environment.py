from gym_super_mario_bros import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import gym
from reward.wrapper import CustomMarioEnvRewardWrapper
from reward.custom_rewards import custom_reward, print_reward

def _register_custom_mario_env(id, **kwargs):
    """
    Register a Super Mario Bros. (1/2) environment with OpenAI Gym.

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
        self._repetition_count = 0

    @property
    def _time_penalty(self):
        """Return the reward for the in-game clock ticking."""
        #_reward = self._time - self._time_last
        self._time_last = self._time
        _reward = -0.05
        return _reward
    
    @property
    def _repetitive_action_penalty(self):
        """Penalty for repeating the same action without moving."""
        # Penalize repetitive actions only if stuck
        if self._action == self._last_action:
            self._repetition_count += 1
        else:
            self._repetition_count = 0

        self._last_action = self._action

        # Apply a penalty only if stuck and repeating an action too long
        if self._repetition_count > 5 and abs(self._x_position - self._x_position_last) < 3:
            print(f"Repetitive action penalty: -1 (count: {self._repetition_count})")
            return -1  # Penalize for repetitive non-progressive actions
        return 0

    def _get_reward(self):
        """Return the reward after a step occurs."""
        return super()._get_reward() + self._repetitive_action_penalty
        # # Original reward calculation
        # base_reward = self._x_reward + self._time_penalty + self._death_penalty
        # # Add the repetitive action penalty
        # total_reward = base_reward + self._repetitive_action_penalty
        # print(f"Base reward: {base_reward}, Repetitive penalty: {self._repetitive_action_penalty}, Total reward: {total_reward}")
        # return total_reward


def create_env():
    env = gym.make('CustomSuperMarioBros-v0')
    # env = CustomMarioEnvRewardWrapper(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def create_training_env():
    env = create_env()
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env
