import gymnasium as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv

import gym_super_mario_bros

def register_custom_mario_env(id, **kwargs):
    """
    Register a Super Mario Bros environment with OpenAI Gym.
    """
    gym.envs.registration.register(
        id=id,
        entry_point='custom_env:CustomMarioEnv',
        max_episode_steps=9999999,
        reward_threshold=9999999,
        kwargs=kwargs,
        nondeterministic=True,
    )

def register_all():
    register_custom_mario_env('CustomSuperMarioBros-v0', rom_mode='vanilla')
    register_custom_mario_env('CustomSuperMarioBros-v1', rom_mode='downsample')
    register_custom_mario_env('CustomSuperMarioBros-v2', rom_mode='pixel')
    register_custom_mario_env('CustomSuperMarioBros-v3', rom_mode='rectangle')

def create_env(version='v0', render_mode=None):
    #env = gym.make('CustomSuperMarioBros-' + version)
    env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode=render_mode)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayscaleObservation(env, keep_dim=True)
    # We can resize the observation, to reduce the computation
    # env = ResizeObservation(env, (60, 64))
    return env

def create_training_env(version='v0', render_mode=None):
    env = create_env(version, render_mode=render_mode)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env

def create_parallel_training_env(version='v0', render_mode=None, parallel=2):
    env = SubprocVecEnv([
        lambda: create_env(version=version, render_mode=render_mode) for _ in range(parallel)
    ])
    return env

register_all()