"""Attack output must run through P1's featurize (called AFTER any attack) on PaySim- and ULB-shaped windows."""
import numpy as np, pandas as pd, pytest
from src.attacks import apply_attack
from src.pipeline.data import featurize
from tests.attacks.test_adapter import CONFIG, make

NAMES = ["random_flip", "targeted_flip", "feature_shift", "feature_scale", "feature_noise", "backdoor"]
LABEL_ONLY = ["random_flip", "targeted_flip"]


@pytest.mark.parametrize("timing", ["uniform", "burst", "low_and_slow"])
@pytest.mark.parametrize("name", NAMES)
def test_paysim_window_featurizes(name, timing):
    df = make()
    out, m = apply_attack(df, name, 0.5, timing, 0, "paysim", CONFIG)
    win = out[(out["step"] >= 100) & (out["step"] < 124)]                       # one 24-step window
    Xw = featurize(win, "paysim")
    assert Xw.shape == (len(win), 12) and Xw.dtype == float and np.isfinite(Xw).all()
    Xc, Xp = featurize(df, "paysim"), featurize(out, "paysim")
    assert np.isfinite(Xp).all()
    assert np.array_equal(Xc[~m], Xp[~m])                                        # untouched rows: identical features
    if name in LABEL_ONLY:
        assert np.array_equal(Xc, Xp)                                            # label attacks never change features
    else:
        assert (Xc[m] != Xp[m]).any()


def test_ulb_window_featurizes():
    r = np.random.default_rng(0); n = 600
    df = pd.DataFrame(r.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = r.lognormal(3, 1, n)
    df.insert(0, "record_id", np.arange(n)); df.insert(1, "step", np.arange(n) // 10)
    df["isFraud"] = (r.random(n) < 0.1).astype(int)
    cfg = {"datasets": {"ulb": {"test_from_step": 36, "window_steps": 2}},
           "attacks": {**CONFIG["attacks"], "features": {"ulb": [f"V{i}" for i in range(1, 29)] + ["Amount"]},
                       "trigger": {"ulb": {"V1": -6.0, "V2": 6.0}}}}
    for name in NAMES:
        out, m = apply_attack(df, name, 0.5, "uniform", 0, "ulb", cfg)
        X = featurize(out[(out["step"] >= 10) & (out["step"] < 12)], "ulb")
        assert X.shape[1] == 29 and np.isfinite(X).all() and m.sum() > 0
