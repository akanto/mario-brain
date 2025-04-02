
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule

from env_factory import create_training_env, create_parallel_training_env

import argparse
import os
import torch
from location import MODEL_DIR, LOG_DIR, MODEL_PATH, LATEST_MODEL_PATH

from stable_baselines3.common.callbacks import CallbackList
from callbacks.tensor_callback import TensorboardCallback
from callbacks.checkpoint_callback import CheckpointCallback

def callback_list() -> CallbackList:
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path=MODEL_DIR,
        verbose=1
    )
    tensorboard_callback = TensorboardCallback(verbose=1)

    return CallbackList([checkpoint_callback, tensorboard_callback])


# On some systems, e.g MacOS, the default device is 'cpu' but we can use 'mps' for better performance https://developer.apple.com/metal/pytorch/
def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def linear_schedule(initial_value: float, min_value: float) -> Schedule:
    def func(progress_remaining: float) -> float:
        return max(min_value, progress_remaining * initial_value)
    return func

def init(total_timesteps, parallel=8):
    torch.set_num_threads(parallel)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    #env = create_training_env('v0')
    env = create_parallel_training_env(version='v0', parallel=parallel, render_mode='rgb_array', record_video=False, total_timesteps=total_timesteps)
    device = get_torch_device()
    print(f"Training on {device}")
    return env, device

def get_model(env, device):
    new_model = not os.path.exists(LATEST_MODEL_PATH + ".zip")
    if new_model:
        print("Training from scratch, creating new model")
        model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, 
                    learning_rate=linear_schedule(0.00001, 0.000001), ent_coef=0.02, batch_size=2048, n_steps=1024, device=device)
    else:
        print(f"Loading model from {LATEST_MODEL_PATH}.zip")
        model = PPO.load(path=LATEST_MODEL_PATH, env=env, device=device)
    return model, new_model

def train(parallel: int = 8, total_timesteps: int = 50_000_000):
    env, device = init(total_timesteps, parallel)

    env.reset()

    model, reset_num_timesteps = get_model(env, device)
    print(f"Number of timesteps trained: {model.num_timesteps}, reset_num_timesteps: {reset_num_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=callback_list(), reset_num_timesteps=reset_num_timesteps)
    model.save(MODEL_PATH)

    print(f"Training complete, model saved to {MODEL_PATH}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total timesteps to train")
    args = parser.parse_args()

    train(parallel=args.parallel, total_timesteps=args.timesteps)