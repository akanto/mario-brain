from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

terminated = True
for step in range(5000):
    if terminated:
        obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()

env.close()