# from stable_baselines3.common.type_aliases import Schedule

# def capped_linear_schedule(initial_value: float, min_value: float) -> Schedule:
#     def func(progress_remaining: float) -> float:
#         return max(min_value, progress_remaining * initial_value)
#     return func

class CappedLinearSchedule:
    def __init__(self, initial_value: float, min_value: float):
        self.initial_value = initial_value
        self.min_value = min_value

    def __call__(self, progress_remaining: float) -> float:
        return max(self.min_value, self.initial_value * progress_remaining)
