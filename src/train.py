from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule

from utils import ensure_directory_exists
from environment import create_training_env

import torch
import location

def linear_schedule(initial_value: float) -> Schedule:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train():

    ensure_directory_exists(location.MODEL_DIR)
    ensure_directory_exists(location.LOG_DIR)

    # On some systems, e.g MacOS, the default device is 'cpu' but we can use 'mps' for better performance https://developer.apple.com/metal/pytorch/
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    env = create_training_env()

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=location.LOG_DIR, learning_rate=0.00001, n_steps=512, device=device)
    model.learn(total_timesteps=1000000)
    model.save(location.MODEL_PATH)

    print(f"Training complete, model saved to {location.MODEL_PATH}")

    env.close()

if __name__ == "__main__":
    train()