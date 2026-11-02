"""Fraud-label alarm: detects a flood of fraud labels (e.g. a random label-flip attack).

Fraud is rare (about 26 labelled frauds per 24-step PaySim window). A label-flip attack that turns even
5% of the legit rows into "fraud" multiplies that count by 50-100x. A reference built from accepted rows
would then be dominated by flipped records (the poison outnumbers the real minority class), so the
defense would treat the real frauds as the outliers and quarantine them. The alarm lets the defense
freeze its minority-class references while the count is abnormal.

Why a COUNT per window and not a rate: windows span a fixed amount of event time and real frauds arrive at a
roughly constant pace, while transaction volume swings a lot (a quiet night window has 93 rows and 23 frauds, a
25% rate). A rate test false-alarms on those windows. The count is stable.

Test: one-sided Poisson z-test of the window's positive count against the mean count of trusted and
alarm-free windows, plus a minimum ratio so that small absolute changes do not alarm.
"""
import numpy as np


class FraudRateAlarm:
    def __init__(self, z=6.0, min_ratio=3.0, min_rows=0):
        self.z, self.min_ratio, self.min_rows = z, min_ratio, min_rows
        self.windows = 0   # baseline windows
        self.k = 0         # baseline positives
        self.history = []

    @property
    def baseline(self):
        """Mean number of positives per baseline window."""
        return self.k / self.windows if self.windows else None

    def observe(self, n_rows: int, n_pos: int, trusted: bool = False) -> bool:
        """Return True if this window holds abnormally many positives. The baseline only learns from
        trusted windows and from windows that did not alarm (so a flood cannot redefine 'normal')."""
        alarm = False
        if not trusted and self.windows > 0 and n_rows >= self.min_rows:
            lam = max(self.baseline, 1.0)
            alarm = bool((n_pos - lam) / np.sqrt(lam) > self.z and n_pos > self.min_ratio * lam)
        self.history.append({"n": n_rows, "pos": n_pos, "alarm": alarm})
        if not alarm:
            self.windows += 1
            self.k += n_pos
        return alarm
