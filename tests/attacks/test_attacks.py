import numpy as np, pandas as pd, pytest
from attacks import attack

def make(n=1000, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame(r.normal(size=(n, 3)), columns=["V1", "V2", "V3"])
    df.insert(0, "record_id", np.arange(n))
    df.insert(1, "event_time", np.arange(n, dtype=float))
    df["label"] = (r.random(n) < 0.1).astype(int)
    return df

CFGS = [
    {"type": "random_label_flip", "rate": 0.10},
    {"type": "targeted_label_flip", "rate": 0.5},
    {"type": "feature_poison", "rate": 0.1, "mode": "shift", "strength": 3},
    {"type": "feature_poison", "rate": 0.1, "mode": "scale", "strength": 5},
    {"type": "feature_poison", "rate": 0.1, "mode": "noise", "strength": 2},
]

@pytest.mark.parametrize("cfg", CFGS)
def test_reproducible_and_pure(cfg):
    b = make(); before = b.copy()
    o1, m1 = attack(b, cfg, np.random.default_rng(7))
    o2, m2 = attack(b, cfg, np.random.default_rng(7))
    assert m1.equals(m2) if hasattr(m1, "equals") else (m1 == m2).all()
    pd.testing.assert_frame_equal(o1, o2)
    pd.testing.assert_frame_equal(b, before)          # input untouched
    assert (o1["is_poisoned"].to_numpy() == m1).all()

@pytest.mark.parametrize("cfg", CFGS)
def test_only_masked_rows_change(cfg):
    b = make()
    o, m = attack(b, cfg, np.random.default_rng(1))
    cols = [c for c in b.columns]
    diff = (o[cols].to_numpy() != b[cols].to_numpy()).any(axis=1)
    assert not diff[~m].any()
    assert diff[m].all()

def test_random_flip_rate():
    b = make()
    o, m = attack(b, {"type": "random_label_flip", "rate": 0.2}, np.random.default_rng(0))
    assert m.sum() == 200
    assert (o["label"][m] != b["label"][m]).all()

def test_targeted_only_fraud_to_legit():
    b = make()
    o, m = attack(b, {"type": "targeted_label_flip", "rate": 1.0}, np.random.default_rng(0))
    assert m.sum() == (b["label"] == 1).sum()
    assert (b["label"][m] == 1).all() and (o["label"][m] == 0).all()

def test_feature_poison_keeps_labels():
    b = make()
    o, _ = attack(b, CFGS[2], np.random.default_rng(0))
    assert (o["label"] == b["label"]).all()

def test_zero_rate_empty_mask():
    _, m = attack(make(), {"type": "random_label_flip", "rate": 0.0}, np.random.default_rng(0))
    assert m.sum() == 0
