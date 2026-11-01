"""Checks the logic the Kafka consumer relies on, without needing a broker:
JSON round trip + shuffled arrival order + type restore must reproduce the offline run."""
import json
import random
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml

from src.pipeline.core import StreamRun, prepare_run, records_to_window
from src.pipeline.replay import replay_windows
from src.pipeline.run_offline import run_offline
from tests.test_harness import fake_df

CFG = yaml.safe_load(open("config/default.yaml"))


@pytest.fixture
def fake_modules(monkeypatch):
    """Stand-ins for P2's src.attacks.api and P3's src.defense.pipeline."""
    api = types.ModuleType("src.attacks.api")

    def apply_attack(df, attack_name, budget, timing, seed, dataset):
        df = df.sort_values(["step", "record_id"], kind="stable").reset_index(drop=True)
        rng = np.random.default_rng(seed)
        cand = np.flatnonzero(df["isFraud"].to_numpy() == 1)
        n = int(round(budget * len(cand)))
        pos = cand[:n] if timing == "burst" else np.sort(rng.choice(cand, n, replace=False))
        mask = np.zeros(len(df), dtype=bool)
        mask[pos] = True
        df.loc[pos, "isFraud"] = 0
        if attack_name == "backdoor":
            df.loc[pos, "oldbalanceDest"] = 1234.56
        return df, mask

    def make_triggered_test(fraud_df, dataset):
        out = fraud_df.copy()
        out["oldbalanceDest"] = 1234.56
        return out

    api.apply_attack, api.make_triggered_test = apply_attack, make_triggered_test
    dfn = types.ModuleType("src.defense.pipeline")

    def defend(window_df, state):
        q = (window_df["oldbalanceDest"] == 1234.56) | (window_df["record_id"] % 11 == 0)
        return window_df[~q], window_df[q], {"seen": state.get("seen", 0) + len(window_df)}

    dfn.defend = defend
    monkeypatch.setitem(sys.modules, "src.attacks.api", api)
    monkeypatch.setitem(sys.modules, "src.defense.pipeline", dfn)


def test_offline_with_attack_and_defense_fills_new_columns(fake_modules):
    out = run_offline("paysim", "backdoor", 0.5, "burst", "on", 0, CFG, data=fake_df())
    assert out["backdoor_hit_rate"].notna().all()
    assert out["det_fpr"].notna().any()
    assert out["time_to_detect"].nunique() == 1 and out["time_to_detect"].notna().all()


@pytest.mark.parametrize("attack,timing,defense", [
    ("none", "uniform", "off"),
    ("targeted_flip", "uniform", "on"),
    ("backdoor", "burst", "on"),
])
def test_kafka_style_transport_matches_offline(fake_modules, attack, timing, defense):
    df = fake_df()
    off = run_offline("paysim", attack, 0.5, timing, defense, 0, CFG, data=df)

    poisoned, test, dcfg = prepare_run("paysim", attack, 0.5, timing, 0, CFG, data=df)
    run = StreamRun("paysim", test, CFG, attack, 0.5, timing, defense, 0, "kafka")
    dtypes = poisoned.dtypes.to_dict()
    rnd = random.Random(0)
    for k, win in replay_windows(poisoned, dcfg["window_steps"]):
        recs = [json.loads(json.dumps(dict(r, _w=k))) for r in win.to_dict("records")]
        rnd.shuffle(recs)                              # partitions interleave arbitrarily
        run.step(k, records_to_window(recs, dtypes))
    kaf = run.result()

    cols = ["window", "n_seen", "n_quarantined", "pr_auc", "recall", "f1", "det_precision",
            "det_recall", "det_fpr", "time_to_detect", "backdoor_hit_rate"]
    assert np.allclose(kaf[cols].to_numpy(float), off[cols].to_numpy(float),
                       equal_nan=True, atol=1e-9)
    if defense == "on":
        assert kaf["det_fpr"].notna().any()
    if attack == "backdoor":
        assert kaf["backdoor_hit_rate"].notna().all()