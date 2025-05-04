from gym_super_mario_bros import SuperMarioBrosEnv


class CustomMarioEnv(SuperMarioBrosEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action = None
        self._last_action_repetition_count = 0

    @property
    def _time_penalty(self):
        """Return the reward for the in-game clock ticking."""
        self._time_last = self._time
        # Overwrite the time penalty with a small negative number for each step
        _reward = -0.05
        return _reward

    @property
    def _repetitive_action_penalty(self):
        """Penalty for repeating the same action without moving."""
        # Apply a penalty only if stuck and repeating an action too long
        if self._last_action_repetition_count > 5 and self._x_reward <= 0:
            # print(f"Repetitive action penalty: -1 (count: {self._last_action_repetition_count})")
            return -1  # Penalize for repetitive non-progressive actions
        return 0

    def _register_action(self, action):
        if self._last_action == action:
            self._last_action_repetition_count += 1
        else:
            self._last_action_repetition_count = 0
        self._last_action = action

    def step(self, action):
        """Override the step method to capture the action taken."""
        self._register_action(action)
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def _get_reward(self):
        """Return the reward after a step occurs."""
        _reward = super()._get_reward() + self._repetitive_action_penalty
        # print(f"Total reward: {_reward}")
        return _reward
