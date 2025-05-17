from mario_brain.schedules.linear_schedule import CappedLinearSchedule


def test_schedule_above_min_value():
    schedule = CappedLinearSchedule(initial_value=1.0, min_value=0.1)
    assert schedule(0.5) == 0.5


def test_schedule_at_min_value():
    schedule = CappedLinearSchedule(initial_value=1.0, min_value=0.1)
    assert schedule(0.05) == 0.1


def test_schedule_at_initial_value():
    schedule = CappedLinearSchedule(initial_value=1.0, min_value=0.1)
    assert schedule(1.0) == 1.0


def test_schedule_below_min_value():
    schedule = CappedLinearSchedule(initial_value=1.0, min_value=0.1)
    assert schedule(0.0) == 0.1
