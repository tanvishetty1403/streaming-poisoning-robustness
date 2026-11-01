import numpy as np, pandas as pd, pytest
from src.attacks import run_attack, apply_attack, make_triggered_test
from tests.attacks.test_adapter import CONFIG, make, FLOATS

TRIG = {"V1": 9.0, "V2": -9.0}


def cmake(n=1000):
    r = np.random.default_rng(0)
    df = pd.DataFrame(r.normal(size=(n, 3)), columns=["V1", "V2", "V3"])
    df.insert(0, "record_id", np.arange(n)); df.insert(1, "step", np.arange(n))
    df["isFraud"] = (r.random(n) < 0.1).astype(int)
    return df


def bd(**kw):
    return {"type": "backdoor", "rate": 0.5, "trigger": TRIG, **kw}


def test_backdoor_stamps_trigger_and_relabels_fraud():
    b = cmake()
    o, m = run_attack(b, bd(), np.random.default_rng(0))
    fraud = int((b["isFraud"] == 1).sum())
    assert abs(m.sum() - 0.5 * fraud) <= 1 and (b["isFraud"][m] == 1).all()
    assert (o["isFraud"][m] == 0).all() and (o["V1"][m] == 9.0).all() and (o["V2"][m] == -9.0).all()
    assert np.array_equal(o[b.columns].ne(b).any(axis=1).to_numpy(), m)       # only masked rows changed
    assert (o["V3"] == b["V3"]).all()


def test_backdoor_int_column_keeps_dtype():
    b = cmake(); b["cnt"] = np.random.default_rng(1).integers(1, 50, len(b))
    o, m = run_attack(b, bd(trigger={"cnt": 7.4}), np.random.default_rng(0))
    assert o["cnt"].dtype == b["cnt"].dtype and (o["cnt"][m] == 7).all() and (o["cnt"][~m] == b["cnt"][~m]).all()


@pytest.mark.parametrize("trig", [None, {}, [], {"V1": "x"}, {1: 2.0}, {"V1": True}])
def test_backdoor_bad_trigger(trig):
    with pytest.raises(ValueError):
        run_attack(cmake(), bd(trigger=trig), np.random.default_rng(0))


def test_backdoor_trigger_column_missing_or_non_numeric():
    b = cmake(); b["merchant"] = "a"
    for trig in ({"nope": 1.0}, {"merchant": 1.0}):
        with pytest.raises(ValueError):
            run_attack(b, bd(trigger=trig), np.random.default_rng(0))


@pytest.mark.parametrize("timing", ["uniform", "burst", "low_and_slow"])
def test_contract_apply_attack_all_names(timing):
    df = make()
    for name in ["random_flip", "targeted_flip", "feature_shift", "feature_scale", "feature_noise", "backdoor"]:
        out, m = apply_attack(df, name, 0.2, timing, 0, "paysim", CONFIG)
        assert m.dtype == bool and len(m) == len(df) and "is_poisoned" not in out and m.sum() > 0
        assert out.dtypes.equals(df.dtypes) and (out["record_id"] == df["record_id"]).all()


def test_apply_attack_backdoor_trigger_from_config():
    df = make()
    out, m = apply_attack(df, "backdoor", 0.5, "uniform", 0, "paysim", CONFIG)
    assert (out["amount"][m] == 9999.99).all() and (out["oldbalanceOrg"][m] == 9999.99).all()
    assert (out["isFraud"][m] == 0).all() and (df["isFraud"][m] == 1).all()


def test_apply_attack_unknown_dataset_or_name():
    with pytest.raises(ValueError):
        apply_attack(make(), "random_flip", 0.1, "uniform", 0, "nope", CONFIG)
    with pytest.raises(ValueError):
        apply_attack(make(), "nope", 0.1, "uniform", 0, "paysim", CONFIG)


def test_make_triggered_test():
    df = make(); fraud = df[df["isFraud"] == 1].copy(); before = fraud.copy()
    t = make_triggered_test(fraud, "paysim", CONFIG)
    pd.testing.assert_frame_equal(fraud, before)                               # input untouched
    assert (t["amount"] == 9999.99).all() and (t["oldbalanceOrg"] == 9999.99).all()
    assert (t["isFraud"] == 1).all() and t.dtypes.equals(fraud.dtypes)
    assert (t["newbalanceOrig"] == fraud["newbalanceOrig"]).all() and len(t) == len(fraud)
    with pytest.raises(ValueError):
        make_triggered_test(fraud, "ulb", CONFIG)
