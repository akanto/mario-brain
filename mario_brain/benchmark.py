import torch
import torch.profiler
from mario_brain.env_factory import create_env, create_training_env
from stable_baselines3 import PPO
import time
import numpy as np
import torch
import time
from mario_brain.location import MODEL_PATH
from mario_brain.train import get_torch_device

def transfer_speed():
    """
    Data Transfer Speed: How fast NumPy data can be moved from CPU memory to GPU memory.
    """
    obs = np.random.rand(4, 240, 256).astype(np.float32)

    device = get_torch_device()

    start_time = time.time()
    for _ in range(10000):
        tensor = torch.tensor(obs, device=device)
    print(f"Data transfer speed between cpu and {device}: {10000 / (time.time() - start_time)} FPS")

def environment_speed():
    env = create_env('v3')
    env.reset()

    start_time = time.time()
    for _ in range(1000):
        env.step(env.action_space.sample())
    print(f"Environment speed (with random actions): {1000 / (time.time() - start_time)} FPS")

def ppo_model_inference_speed():
    env = create_training_env('v3')
    env.reset()

    model = PPO.load(path=MODEL_PATH, env=env)

    obs = env.reset()
    start_time = time.time()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
    print(f"Inference speed on PPO model: {1000 / (time.time() - start_time)} FPS")

transfer_speed()
environment_speed()
ppo_model_inference_speed()