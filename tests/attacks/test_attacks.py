import numpy as np, pandas as pd, pytest
from attacks import attack

FEATS = ["V1", "V2", "V3"]


def make(n=1000, seed=0, fraud=0.1):
    r = np.random.default_rng(seed)
    df = pd.DataFrame(r.normal(size=(n, 3)), columns=FEATS)
    df.insert(0, "record_id", np.arange(n))
    df.insert(1, "event_time", np.arange(n, dtype=float))
    df["is_online"] = r.integers(0, 2, n)          # binary column, never poisoned
    df["merchant"] = r.choice(["a", "b", "c"], n)  # string column, never poisoned
    df["label"] = (r.random(n) < fraud).astype(int)
    return df


STD = {c: 1.0 for c in FEATS}


def fp(**kw):
    cfg = {"type": "feature_poison", "rate": 0.1, "mode": "shift", "strength": 3,
           "features": FEATS, "feature_std": STD}
    cfg.update(kw)
    return cfg


CFGS = [
    {"type": "random_label_flip", "rate": 0.10},
    {"type": "targeted_label_flip", "rate": 0.5},
    fp(mode="shift"), fp(mode="scale", strength=5), fp(mode="noise", strength=2),
]


@pytest.mark.parametrize("cfg", CFGS)
def test_reproducible_and_pure(cfg):
    b = make(); before = b.copy()
    o1, m1 = attack(b, cfg, np.random.default_rng(7))
    o2, m2 = attack(b, cfg, np.random.default_rng(7))
    assert np.array_equal(m1, m2)
    pd.testing.assert_frame_equal(o1, o2)
    pd.testing.assert_frame_equal(b, before)
    assert np.array_equal(o1["is_poisoned"].to_numpy(), m1)


@pytest.mark.parametrize("cfg", CFGS)
def test_mask_equals_rows_that_changed(cfg):
    b = make()
    o, m = attack(b, cfg, np.random.default_rng(1))
    changed = o[b.columns].ne(b).any(axis=1).to_numpy()
    assert np.array_equal(changed, m)


def test_random_flip_rate():
    b = make()
    o, m = attack(b, {"type": "random_label_flip", "rate": 0.2}, np.random.default_rng(0))
    assert abs(m.sum() - 200) <= 1
    assert (o["label"][m] != b["label"][m]).all()


def test_targeted_only_fraud_to_legit():
    b = make()
    o, m = attack(b, {"type": "targeted_label_flip", "rate": 1.0}, np.random.default_rng(0))
    assert m.sum() == (b["label"] == 1).sum()
    assert (b["label"][m] == 1).all() and (o["label"][m] == 0).all()


def test_zero_rate_empty_mask():
    _, m = attack(make(), {"type": "random_label_flip", "rate": 0.0}, np.random.default_rng(0))
    assert m.sum() == 0


def test_small_batch_stochastic_rounding_013pct_fraud():
    b = make(n=10000); b["label"] = 0; b.loc[:12, "label"] = 1     # 13 fraud = 0.13%
    cfg = {"type": "targeted_label_flip", "rate": 0.05}             # expect 0.65 flips
    counts = [attack(b, cfg, np.random.default_rng(s))[1].sum() for s in range(2000)]
    assert max(counts) == 1 and abs(np.mean(counts) - 0.65) < 0.06  # old round() gave 0 always
    b2 = make(n=50)
    cfg2 = {"type": "random_label_flip", "rate": 0.01}              # expect 0.5 flips
    c2 = [attack(b2, cfg2, np.random.default_rng(s))[1].sum() for s in range(2000)]
    assert abs(np.mean(c2) - 0.5) < 0.06


def test_constant_column_not_marked():
    b = make(); b["C"] = 5.0
    o, m = attack(b, fp(rate=1.0, features=["C"], feature_std={"C": 0.0}), np.random.default_rng(0))
    assert m.sum() == 0 and o["is_poisoned"].sum() == 0 and (o["C"] == 5.0).all()
    o, m = attack(b, fp(rate=1.0, mode="scale", strength=1, features=["C"]), np.random.default_rng(0))
    assert m.sum() == 0


def test_shift_uses_reference_std_not_batch_std():
    b = make()
    ref = {c: 10.0 for c in FEATS}                      # batch std is ~1
    o, m = attack(b, fp(feature_std=ref, strength=3), np.random.default_rng(0))
    assert m.sum() > 0
    assert np.allclose(o[FEATS][m].to_numpy() - b[FEATS][m].to_numpy(), 30.0)


def test_scale_effect_size():
    b = make()
    o, m = attack(b, fp(mode="scale", strength=5), np.random.default_rng(0))
    assert np.allclose(o[FEATS][m].to_numpy(), 5 * b[FEATS][m].to_numpy())


def test_noise_effect_size():
    b = make(n=20000)
    o, m = attack(b, fp(mode="noise", strength=1.5, rate=1.0, feature_std={c: 2.0 for c in FEATS}),
                  np.random.default_rng(0))
    assert abs((o[FEATS].to_numpy() - b[FEATS].to_numpy()).std() - 3.0) < 0.1


def test_feature_subset_only():
    b = make()
    o, m = attack(b, fp(features=["V1"]), np.random.default_rng(0))
    assert m.sum() > 0
    for c in ["V2", "V3", "is_online", "merchant", "label"]:
        assert (o[c] == b[c]).all()
    assert (o["V1"][m] != b["V1"][m]).all()


def test_target_label():
    b = make()
    o, m = attack(b, fp(rate=0.5, target_label=1), np.random.default_rng(0))
    assert m.sum() > 0 and (b["label"][m] == 1).all()


def _bad_cfgs():
    no_feats = {k: v for k, v in fp().items() if k != "features"}
    no_std = {k: v for k, v in fp().items() if k != "feature_std"}
    no_type = {"rate": 0.1}
    no_rate = {"type": "random_label_flip"}
    return [
        {"type": "nope", "rate": 0.1}, no_type, no_rate,
        {"type": "random_label_flip", "rate": -0.1},
        {"type": "random_label_flip", "rate": 1.5},
        {"type": "random_label_flip", "rate": "0.1"},
        fp(mode="bogus", rate=0.0),                      # checked even when rate is 0
        fp(strength="x"), no_feats, no_std, fp(target_label=2),
    ]


@pytest.mark.parametrize("cfg", _bad_cfgs())
def test_validation_errors(cfg):
    with pytest.raises(ValueError):
        attack(make(), cfg, np.random.default_rng(0))


def test_feature_not_in_batch():
    with pytest.raises(ValueError):
        attack(make(), fp(features=["nope"], feature_std={"nope": 1.0}), np.random.default_rng(0))


@pytest.mark.parametrize("second", [{"type": "random_label_flip", "rate": 0.2},
                                    {"type": "targeted_label_flip", "rate": 0.5}])
def test_stacked_attacks_accumulate(second):
    b = make()
    o1, m1 = attack(b, fp(rate=0.2), np.random.default_rng(1))
    o2, m2 = attack(o1, second, np.random.default_rng(2))
    assert np.array_equal(o2["is_poisoned"].to_numpy(), m1 | m2)
    assert o2["is_poisoned"][m1].all()                   # second attack didn't erase the first
    o3, m3 = attack(*attack(b, second, np.random.default_rng(2))[:1], fp(rate=0.2), np.random.default_rng(1))
    assert o3["is_poisoned"].sum() >= m3.sum()
