import numpy as np
import pandas as pd
from helpers import make_clean
from test_defense import cfgx, run
from src.defense import make_state
from src.defense.detectors import Mahalanobis
from src.defense.metrics import detection_metrics


def correlated(n, seed, start=0, rho=0.95):
    r = np.random.default_rng(seed)
    a = r.normal(size=n)
    b = rho * a + np.sqrt(1 - rho ** 2) * r.normal(size=n)
    df = make_clean(n, d=3, seed=seed, start=start)
    df["V1"], df["V2"] = a, b
    return df


def break_correlation(df, frac, seed=0):
    """Flip V2's sign on rows where |V1| > 1: each value is still ordinary alone, the pair is impossible."""
    r = np.random.default_rng(seed)
    out = df.copy()
    cand = np.flatnonzero((out["V1"].abs() > 1.0).to_numpy())
    idx = r.choice(cand, size=int(frac * len(cand)), replace=False)
    mask = np.zeros(len(out), dtype=bool)
    mask[idx] = True
    out.iloc[idx, out.columns.get_loc("V2")] = -out.iloc[idx]["V2"].to_numpy()
    return out, mask


def state(dets, **kw):
    return make_state(cfgx(detectors=dets, class_conditional=False, trusted_windows=1, **kw))


def test_mahalanobis_catches_what_robust_z_misses():
    res = {}
    for dets in (["robust_z"], ["mahalanobis"]):
        st = state(dets)
        run(st, correlated(3000, 0))
        w, mask = break_correlation(correlated(3000, 1, start=3000), 0.5)
        _, _, _, flagged = run(st, w)
        res[dets[0]] = detection_metrics(flagged, mask)
    print({k: {m: round(v[m], 3) for m in ("precision", "recall", "fpr")} for k, v in res.items()})
    assert res["mahalanobis"]["recall"] >= 0.9 and res["mahalanobis"]["fpr"] <= 0.01
    assert res["robust_z"]["recall"] < 0.2


def test_clean_window_false_positive_rate_is_low():
    st = state(["mahalanobis"])
    run(st, correlated(3000, 0))
    _, quar, _, flagged = run(st, correlated(3000, 5, start=3000))
    assert flagged.mean() <= 0.005


def test_silent_until_reference_exists():
    d = Mahalanobis(1.0, 5000, 200, class_conditional=False)
    X = np.random.default_rng(0).normal(size=(100, 3))
    assert not d.ready and (d.score(X) == 0).all()


def test_constant_columns_are_ignored():
    d = Mahalanobis(1.0, 5000, 200, class_conditional=False)
    R = np.random.default_rng(0).normal(size=(1000, 3)); R[:, 2] = 7.0
    d.update(R)
    Y = R[:50].copy(); Y[:, 2] = 9.0                     # a change in a constant column is not scored
    assert np.isfinite(d.score(Y)).all() and (d.score(Y) < 1).all()


def test_quarantined_rows_do_not_enter_the_reference():
    st = state(["mahalanobis"], maha_window=100000)
    run(st, correlated(3000, 0))
    n0 = len(st["detectors"][0]._ref[None])
    w, mask = break_correlation(correlated(3000, 1, start=3000), 1.0)
    _, quar, _, _ = run(st, w)
    assert len(st["detectors"][0]._ref[None]) == n0 + (len(w) - len(quar))
