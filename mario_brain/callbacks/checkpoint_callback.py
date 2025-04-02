from stable_baselines3.common.callbacks import BaseCallback
import os

class CheckpointCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int = 1_000_000, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"checkpoint_{self.num_timesteps}.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saved model checkpoint to {path}")
        return True