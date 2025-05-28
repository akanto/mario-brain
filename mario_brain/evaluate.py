import argparse
import os

from mario_brain.location import MODEL_PATH, VIDEO_DIR
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

from mario_brain.env_factory import create_training_env


def record_video(env):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = VecVideoRecorder(
        venv=env,
        video_folder=VIDEO_DIR,
        name_prefix="mario-ppo-evaluation",
        record_video_trigger=lambda x: x == 0,
        video_length=2000,
    )
    return env


def evaluate(model_path: str):
    print("Evaluate")

    env = create_training_env(version="v0", render_mode="human", reward_summary=True)

    # env = record_video(env)

    model = PPO.load(
        path=model_path,
        env=env,
        custom_objects=dict(learning_rate=0.00001),
        device="mps",
    )
    # model = PPO.load(path=model_path, env=env, device='cpu')

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        env.render()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to the model to evaluate"
    )
    args = parser.parse_args()
    evaluate(model_path=args.model)
