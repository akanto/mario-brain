# This file contains the custom reward wrapper for the environment.
from gym_super_mario_bros import SuperMarioBrosEnv

class CustomMarioEnvRewardWrapper(SuperMarioBrosEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action = None
        self._repetition_count = 0

    @property
    def _repetitive_action_penalty(self):
        """Penalty for repeating the same action without moving."""
        # Penalize repetitive actions only if stuck
        if self._action == self._last_action:
            self._repetition_count += 1
        else:
            self._repetition_count = 0

        self._last_action = self._action

        # Apply a penalty only if stuck and repeating an action too long
        if self._repetition_count > 5 and abs(self._x_position - self._x_position_last) < 3:
            print(f"Repetitive action penalty: -1 (count: {self._repetition_count})")
            return -1  # Penalize for repetitive non-progressive actions
        return 0

    def _get_reward(self):
        """Return the reward after a step occurs."""
        # Original reward calculation
        base_reward = self._x_reward + self._time_penalty + self._death_penalty
        # Add the repetitive action penalty
        total_reward = base_reward + self._repetitive_action_penalty
        print(f"Base reward: {base_reward}, Repetitive penalty: {self._repetitive_action_penalty}, Total reward: {total_reward}")
        return total_reward
