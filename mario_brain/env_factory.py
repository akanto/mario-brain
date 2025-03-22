import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

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

def create_env():
    register_all()
    env = gym.make('CustomSuperMarioBros-v0')
    #env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def create_training_env():
    env = create_env()
    env = GrayScaleObservation(env, keep_dim=True)
    # env = ResizeObservation(env, shape=64)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env