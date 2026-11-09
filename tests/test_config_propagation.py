from copy import deepcopy

import numpy as np

from src.pipeline.core import StreamRun, prepare_run
from src.pipeline.data import featurize
from tests.test_harness import CFG, fake_df


def test_custom_attack_strength_reaches_preparation():
    cfg = deepcopy(CFG)
    cfg["attacks"]["methods"]["feature_scale"]["strength"] = 7.0
    data = fake_df()
    poisoned, _, _ = prepare_run("paysim", "feature_scale", 1.0, "uniform", 0, cfg, data)
    changed = poisoned["is_poisoned"].to_numpy()
    assert changed.any()
    clean = data[data.step < cfg["datasets"]["paysim"]["test_from_step"]].reset_index(drop=True)
    np.testing.assert_allclose(poisoned.loc[changed, "amount"], clean.loc[changed, "amount"] * 7)


def test_custom_trigger_and_defense_reach_stream_run():
    cfg = deepcopy(CFG)
    cfg["attacks"]["trigger"]["paysim"] = {"amount": 123.0}
    cfg["defense"]["z_threshold"] = 19.0
    cfg["defense"]["trusted_windows"] = 99
    poisoned, test, _ = prepare_run("paysim", "backdoor", 0.2, "uniform", 0, cfg, fake_df())
    run = StreamRun("paysim", test, cfg, "backdoor", 0.2, "uniform", "on", 0, "config-test")
    expected = test[test.isFraud == 1].copy()
    expected["amount"] = 123.0
    np.testing.assert_allclose(run.X_trig, featurize(expected, "paysim"))
    run.step(0, poisoned[poisoned.step < 25].copy())
    assert run.state["cfg"]["z_threshold"] == 19.0
    assert run.rows[0]["n_quarantined"] == 0
