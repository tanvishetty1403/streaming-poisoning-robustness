import numpy as np, pandas as pd, pytest
from attacks.adapter import attack_stream

FEATS = ["amount", "oldbalanceOrg"]
CONFIG = {
    "datasets": {"paysim": {"test_from_step": 800}},
    "attacks": {
        "label_col": "isFraud", "time_col": "step",
        "features": {"paysim": FEATS},
        "methods": {
            "random_flip": {"type": "random_label_flip"},
            "targeted_flip": {"type": "targeted_label_flip"},
            "feature_shift": {"type": "feature_poison", "mode": "shift", "strength": 3.0},
            "feature_scale": {"type": "feature_poison", "mode": "scale", "strength": 5.0},
            "feature_noise": {"type": "feature_poison", "mode": "noise", "strength": 2.0},
        },
    },
}
NAMES = list(CONFIG["attacks"]["methods"])


def make(n=1000, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"record_id": np.arange(n), "step": np.arange(n),
                       "type": r.choice(["PAYMENT", "TRANSFER"], n),
                       "amount": r.lognormal(5, 1, n), "oldbalanceOrg": r.lognormal(7, 1, n)})
    df["isFraud"] = (r.random(n) < 0.1).astype(int)
    return df


def run(df, name, budget=0.1, seed=0, timing="uniform"):
    return attack_stream(df, name, budget, timing, seed, config=CONFIG)


@pytest.mark.parametrize("name", NAMES)
def test_contract_shape_and_test_set_untouched(name):
    df = make(); before = df.copy()
    out, m = run(df, name)
    pd.testing.assert_frame_equal(df, before)                      # input untouched
    assert isinstance(m, np.ndarray) and m.dtype == bool and len(m) == len(df)
    assert list(out.columns) == list(df.columns) and "is_poisoned" not in out
    assert (out["record_id"] == df["record_id"]).all()             # order kept
    assert m[df["step"] >= 800].sum() == 0                         # no test rows poisoned
    pd.testing.assert_frame_equal(out[df["step"] >= 800], df[df["step"] >= 800])
    assert m.sum() > 0
    assert np.array_equal(out.ne(df).any(axis=1).to_numpy(), m)    # mask == rows changed


@pytest.mark.parametrize("name", NAMES)
def test_seed_reproducible(name):
    df = make()
    (o1, m1), (o2, m2), (_, m3) = run(df, name, seed=3), run(df, name, seed=3), run(df, name, seed=4)
    assert np.array_equal(m1, m2) and o1.equals(o2) and not np.array_equal(m1, m3)


def test_budget_semantics():
    df = make(); train = df[df["step"] < 800]
    _, m = run(df, "random_flip", budget=0.2)
    assert abs(m.sum() - 0.2 * len(train)) <= 1
    out, m = run(df, "targeted_flip", budget=0.5)
    assert abs(m.sum() - 0.5 * train["isFraud"].sum()) <= 1
    assert (df["isFraud"][m] == 1).all() and (out["isFraud"][m] == 0).all()


def test_shift_uses_clean_train_std():
    df = make(); train = df[df["step"] < 800]
    out, m = run(df, "feature_shift")
    want = 3.0 * train[FEATS].std(ddof=0).to_numpy()
    assert np.allclose(out[FEATS][m].to_numpy() - df[FEATS][m].to_numpy(), want)


def test_bad_inputs():
    with pytest.raises(NotImplementedError):
        run(make(), "random_flip", timing="burst")
    with pytest.raises(ValueError):
        run(make(), "nope")
    with pytest.raises(ValueError):
        run(make(), "random_flip", budget=1.5)
