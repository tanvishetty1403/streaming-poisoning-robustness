"""Contract-facing entry point: attack_stream(df, attack_name, budget, timing, seed) -> (df_poisoned, poison_mask).

- Only the TRAIN stream (step < datasets[dataset].test_from_step) is attacked; test rows are never touched.
- budget = fraction of train rows tampered with (targeted_flip: fraction of train fraud rows).
- Params (mode, strength, features, label/time column) come from config/default.yaml -> `attacks:`.
- feature_std = std of the CLEAN train stream (df is the clean stream), never of poisoned data.
- Returns df WITHOUT is_poisoned; the pipeline adds it from poison_mask.
"""
import numpy as np
from . import ATTACKS
from .common import compute_feature_std
from .validate import validate

TIMINGS_IMPLEMENTED = {"uniform"}   # burst / low_and_slow arrive in Phase 3


def load_config(path="config/default.yaml"):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def attack_stream(df, attack_name, budget, timing, seed, dataset="paysim", config=None):
    config = config or load_config()
    acfg = config["attacks"]
    if timing not in TIMINGS_IMPLEMENTED:
        raise NotImplementedError(f"timing {timing!r} not implemented yet (have {sorted(TIMINGS_IMPLEMENTED)})")
    if attack_name not in acfg["methods"]:
        raise ValueError(f"unknown attack {attack_name!r}; known: {sorted(acfg['methods'])}")

    label, tcol = acfg["label_col"], acfg["time_col"]
    train = (df[tcol] < config["datasets"][dataset]["test_from_step"]).to_numpy()

    cfg = dict(acfg["methods"][attack_name], rate=budget, label_col=label)
    changed_cols = [label]
    if cfg["type"] == "feature_poison":
        cfg["features"] = list(acfg["features"][dataset])
        cfg["feature_std"] = compute_feature_std(df[train], cfg["features"])
        changed_cols = cfg["features"]

    validate(cfg)
    sub = df[train].drop(columns=["is_poisoned"], errors="ignore")
    out_sub, m_sub = ATTACKS[cfg["type"]](sub, cfg, np.random.default_rng(seed))

    out = df.drop(columns=["is_poisoned"], errors="ignore").copy()
    for c in changed_cols:
        out.loc[train, c] = out_sub[c].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    mask[np.flatnonzero(train)] = m_sub
    return out, mask
