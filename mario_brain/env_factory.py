from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from reward_summary import CumulativeRewardWrapper

import gym_super_mario_bros

def create_env(version='v0', render_mode=None, reward_summary=False):
    #env = gym.make('CustomSuperMarioBros-' + version)
    env = gym_super_mario_bros.make('SuperMarioBros-' + version, render_mode=render_mode)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayscaleObservation(env, keep_dim=True)
    if reward_summary:
        env = CumulativeRewardWrapper(env)
    # We can resize the observation, to reduce the computation
    # env = ResizeObservation(env, (60, 64))
    return env

def create_training_env(version='v0', render_mode=None, reward_summary=False):
    env = create_env(version, render_mode=render_mode, reward_summary=reward_summary)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env

def create_parallel_training_env(version='v0', render_mode=None, parallel=4, record_video=False, total_timesteps=1000):
    env = SubprocVecEnv([
        lambda: create_env(version=version, render_mode=render_mode) for _ in range(parallel)
    ])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    if record_video:
        env = VecVideoRecorder(env, "videos/",
                           record_video_trigger=lambda step: step == 0,  # Record only the first step
                           video_length=total_timesteps)

    return env