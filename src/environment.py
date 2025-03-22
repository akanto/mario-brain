import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from reward.wrapper import RewardWrapper
from reward.custom_rewards import custom_reward, print_reward

def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    #env = RewardWrapper(env, print_reward)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def create_training_env():
    env = create_env()
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env
