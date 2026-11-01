import numpy as np
import pandas as pd
import yaml
from src.pipeline.replay import replay_windows
from src.pipeline.run_offline import run_offline

CFG = yaml.safe_load(open("config/default.yaml"))


def fake_df(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "step": np.sort(rng.integers(1, 740, n)),
        "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n),
        "amount": rng.gamma(2, 500, n), "oldbalanceOrg": rng.gamma(2, 800, n),
        "newbalanceOrig": rng.gamma(2, 800, n), "oldbalanceDest": rng.gamma(2, 800, n),
        "newbalanceDest": rng.gamma(2, 800, n),
    })
    d["isFraud"] = (rng.random(n) < 0.03).astype(int)
    d.insert(0, "record_id", np.arange(n))
    return d


def test_replay_is_event_time_ordered_and_complete():
    df = fake_df()
    seen, last = [], -1
    for k, w in replay_windows(df, 24):
        assert k > last
        last = k
        seen.append(w)
    out = pd.concat(seen)
    assert len(out) == len(df)
    assert out["step"].is_monotonic_increasing


def test_offline_loop_runs_and_matches_schema():
    out = run_offline("paysim", cfg=CFG, data=fake_df())
    assert {"pr_auc", "recall", "f1", "window"} <= set(out.columns)
    assert len(out) > 10
    assert out["n_seen"].is_monotonic_increasing


def test_same_seed_same_metrics():
    a = run_offline("paysim", seed=1, cfg=CFG, data=fake_df())
    b = run_offline("paysim", seed=1, cfg=CFG, data=fake_df())
    pd.testing.assert_frame_equal(a.drop(columns="run_id"), b.drop(columns="run_id"))