import numpy as np
from src.defense.drift import FraudRateAlarm


def test_quiet_window_with_the_same_fraud_count_does_not_alarm():
    a = FraudRateAlarm()
    a.observe(57000, 24, trusted=True)
    assert not a.observe(93, 23)          # quiet night window: 25% rate, normal count
    assert not a.observe(45000, 31)       # ordinary window


def test_flood_of_fraud_labels_alarms_and_does_not_move_the_baseline():
    a = FraudRateAlarm()
    a.observe(57000, 24, trusted=True)
    before = a.baseline
    assert a.observe(45000, 2377)         # random label flips: ~100x the normal count
    assert a.baseline == before


def test_trusted_windows_never_alarm_and_empty_baseline_is_silent():
    a = FraudRateAlarm()
    assert not a.observe(1000, 500)       # no baseline yet
    assert not FraudRateAlarm().observe(1000, 500, trusted=True)


def test_slow_creep_below_the_ratio_is_not_an_alarm():
    a = FraudRateAlarm()
    a.observe(50000, 25, trusted=True)
    assert not a.observe(50000, 45)       # 1.8x: normal variation for a Poisson count? z=4 -> no alarm
