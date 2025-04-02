from env_factory import create_training_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder

import os
from location import VIDEO_DIR, MODEL_PATH

def record_video(env):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = VecVideoRecorder(venv=env, video_folder=VIDEO_DIR, name_prefix="mario-ppo-evaluation", record_video_trigger=lambda x: x == 0, video_length=2000)
    return env


def evaluate():
    print("Evaluate")

    env = create_training_env(version='v0', render_mode='human', reward_summary=True)
    
    #env = record_video(env)

    model = PPO.load(path=MODEL_PATH, env=env)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
        env.render()
    env.close()

if __name__ == "__main__":
    evaluate()
