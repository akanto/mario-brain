import gym_super_mario_bros
from nes_py.app.play_human import play_human

env = gym_super_mario_bros.make("SuperMarioBros-v0")
play_human(env)
env.close()
