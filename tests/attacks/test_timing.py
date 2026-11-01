import numpy as np, pytest
from src.attacks import attack
from tests.attacks.test_adapter import make

WIN = 24                                   # PaySim window_steps
FLIP = {"type": "random_label_flip"}


def run(df, timing, cfg=FLIP, budget=0.2, seed=0):
    return attack(df, budget, timing, seed, cfg, test_from_step=800)


def per_window(m):
    steps = np.arange(len(m))
    return np.bincount((steps // WIN)[m], minlength=800 // WIN + 1)[: 800 // WIN]   # full train windows


def test_same_budget_every_timing():
    df = make()
    sums = {t: run(df, t)[1].sum() for t in ["uniform", "burst", "low_and_slow"]}
    assert all(abs(v - 160) <= 1 for v in sums.values())


def test_burst_is_one_contiguous_block():
    for s in range(10):
        m = run(make(), "burst", seed=s)[1]
        pos = np.flatnonzero(m)
        assert pos.max() - pos.min() + 1 == len(pos)
        assert len(set(pos // WIN)) <= int(np.ceil(len(pos) / WIN)) + 1       # touches few windows


def test_low_and_slow_is_evenly_spread():
    for s in range(10):
        c = per_window(run(make(), "low_and_slow", seed=s)[1])
        assert c.max() <= int(np.ceil(WIN * 160 / 800)) + 2                  # low per-window rate
        assert c.min() >= 3                                                  # every window gets some


def test_uniform_is_random_spread():
    m = run(make(), "uniform")[1]
    assert (per_window(m) > 0).sum() >= 20 and not (np.diff(np.flatnonzero(m)) == 1).all()


def test_targeted_burst_consecutive_among_fraud():
    df = make(); fraud = np.flatnonzero((df["isFraud"] == 1).to_numpy() & (df["step"] < 800).to_numpy())
    m = run(df, "burst", {"type": "targeted_label_flip"}, budget=0.5)[1]
    ranks = np.flatnonzero(np.isin(fraud, np.flatnonzero(m)))
    assert len(ranks) > 1 and ranks.max() - ranks.min() + 1 == len(ranks)


def test_bad_timing():
    with pytest.raises(ValueError):
        run(make(), "nope")
