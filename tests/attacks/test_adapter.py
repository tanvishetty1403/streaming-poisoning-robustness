import numpy as np, pandas as pd, pytest
from src.attacks import attack, attack_from_config

FEATS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FLOATS = FEATS[:4]
CONFIG = {
    "datasets": {"paysim": {"test_from_step": 800}},
    "attacks": {
        "label_col": "isFraud", "time_col": "step", "features": {"paysim": FEATS},
        "methods": {
            "random_flip": {"type": "random_label_flip"},
            "targeted_flip": {"type": "targeted_label_flip"},
            "feature_shift": {"type": "feature_poison", "mode": "shift", "strength": 3.0},
            "feature_scale": {"type": "feature_poison", "mode": "scale", "strength": 5.0},
            "feature_noise": {"type": "feature_poison", "mode": "noise", "strength": 2.0},
        },
    },
}
NAMES, TIMINGS = list(CONFIG["attacks"]["methods"]), ["uniform", "burst", "low_and_slow"]


def make(n=1000, seed=0):
    """PaySim-shaped frame (raw columns), event-time order."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"record_id": np.arange(n), "step": np.arange(n),
                       "type": r.choice(["PAYMENT", "TRANSFER"], n), "amount": r.lognormal(5, 1, n),
                       "nameOrig": [f"C{i}" for i in range(n)], "oldbalanceOrg": r.lognormal(7, 1, n),
                       "newbalanceOrig": r.lognormal(7, 1, n), "nameDest": [f"M{i}" for i in range(n)],
                       "oldbalanceDest": r.lognormal(6, 1, n),
                       "newbalanceDest": r.lognormal(6, 1, n).round().astype("int64"),   # int feature
                       "isFlaggedFraud": 0})
    df["isFraud"] = (r.random(n) < 0.1).astype(int)
    return df


def run(df, name, budget=0.1, seed=0, timing="uniform"):
    return attack_from_config(df, name, budget, timing, seed, config=CONFIG)


@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("name", NAMES)
def test_contract(name, timing):
    df = make(); before = df.copy()
    out, m = run(df, name, timing=timing)
    pd.testing.assert_frame_equal(df, before)                         # input untouched
    assert isinstance(m, np.ndarray) and m.dtype == bool and len(m) == len(df)
    assert out.dtypes.equals(df.dtypes) and list(out.columns) == list(df.columns)   # schema + dtypes kept
    assert "is_poisoned" not in out and (out["record_id"] == df["record_id"]).all()
    assert m[df["step"] >= 800].sum() == 0                            # no test rows poisoned
    pd.testing.assert_frame_equal(out[df["step"] >= 800], df[df["step"] >= 800])
    assert m.sum() > 0 and np.array_equal(out.ne(df).any(axis=1).to_numpy(), m)


@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("name", NAMES)
def test_seed_reproducible(name, timing):
    df = make()
    (o1, m1), (o2, m2), (_, m3) = run(df, name, seed=3, timing=timing), run(df, name, seed=3, timing=timing), run(df, name, seed=4, timing=timing)
    assert np.array_equal(m1, m2) and o1.equals(o2) and not np.array_equal(m1, m3)


def test_budget_semantics():
    df = make(); train = df[df["step"] < 800]
    _, m = run(df, "random_flip", budget=0.2)
    assert abs(m.sum() - 0.2 * len(train)) <= 1
    out, m = run(df, "targeted_flip", budget=0.5)
    assert abs(m.sum() - 0.5 * train["isFraud"].sum()) <= 1
    assert (df["isFraud"][m] == 1).all() and (out["isFraud"][m] == 0).all()


def test_shift_uses_clean_train_std_and_override():
    df = make(); train = df[df["step"] < 800]
    out, m = run(df, "feature_shift")
    assert np.allclose(out[FLOATS][m].to_numpy() - df[FLOATS][m].to_numpy(), 3.0 * train[FLOATS].std(ddof=0).to_numpy())
    cfg = {"type": "feature_poison", "mode": "shift", "strength": 3.0, "features": FLOATS,
           "feature_std": {c: 10.0 for c in FLOATS}}
    out, m = attack(df, 0.1, "uniform", 0, cfg, test_from_step=800)
    assert np.allclose(out[FLOATS][m].to_numpy() - df[FLOATS][m].to_numpy(), 30.0)


def test_no_test_from_step_attacks_whole_df():
    df = make()
    _, m = attack(df, 0.2, "uniform", 0, {"type": "random_label_flip"})
    assert abs(m.sum() - 200) <= 1 and m[800:].sum() > 0


def test_bad_inputs():
    with pytest.raises(ValueError):
        run(make(), "nope")
    with pytest.raises(ValueError):
        run(make(), "random_flip", timing="sideways")
    with pytest.raises(ValueError):
        run(make(), "random_flip", budget=1.5)
