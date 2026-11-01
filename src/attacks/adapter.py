"""Contract section 2: attack(df, budget, timing, seed) -> (df_poisoned, poison_mask).

cfg (separate attack-config argument): type, mode, strength, features, [feature_std], [target_label],
[label_col='isFraud'], [time_col='step']. Params normally come from config/default.yaml via attack_from_config.
- budget = fraction of the attack's candidate rows tampered with. Contract: random_flip -> all train rows;
  everything else -> fraud rows (config sets target_label: 1 on the feature attacks; backdoor is fraud-only).
- timing: uniform | burst | low_and_slow (see common.pick).
- If test_from_step is given, rows with step >= test_from_step are never touched.
- If clean_until_step is given, rows with step < clean_until_step are never touched either (the trusted
  warm-up windows the defense assumes are clean). attack_from_config sets it to
  min_step + clean_windows * window_steps, where min_step is the smallest step in the stream passed in
  (P1's replay uses window = (step - min_step) // window_steps, so window 0 starts at min_step).
- feature_std defaults to the std of the clean input df (train rows); pass cfg['feature_std'] to override.
- Returns df WITHOUT is_poisoned; the pipeline adds it from poison_mask.
"""
import numpy as np
from .common import compute_feature_std
from .registry import ATTACKS
from .validate import validate


def load_config(path="config/default.yaml"):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def attack(df, budget, timing, seed, cfg, test_from_step=None, clean_until_step=None):
    cfg = dict(cfg)
    label = cfg.setdefault("label_col", "isFraud")
    tcol = cfg.pop("time_col", "step")
    steps = df[tcol].to_numpy()
    train = np.ones(len(df), dtype=bool) if test_from_step is None else steps < test_from_step
    ref = train.copy()                       # all clean pre-test rows: used for feature_std
    if clean_until_step is not None:
        if isinstance(clean_until_step, bool) or not isinstance(clean_until_step, (int, np.integer)) or clean_until_step < 0:
            raise ValueError(f"clean_until_step must be an int >= 0, got {clean_until_step!r}")
        train &= steps >= clean_until_step   # attackable rows
    cfg.update(rate=budget, timing=timing)
    changed_cols = [label]
    if cfg.get("type") == "feature_poison":
        feats = cfg.get("features")
        changed_cols = list(feats) if isinstance(feats, (list, tuple)) else []
        if (cfg.get("mode") in ("shift", "noise") and "feature_std" not in cfg
                and changed_cols and all(f in df.columns for f in changed_cols)):
            cfg["feature_std"] = compute_feature_std(df[ref], changed_cols)
    elif cfg.get("type") == "backdoor":
        trig = cfg.get("trigger")
        changed_cols = (list(trig) if isinstance(trig, dict) else []) + [label]
    validate(cfg)

    sub = df[train].drop(columns=["is_poisoned"], errors="ignore")
    out_sub, m_sub = ATTACKS[cfg["type"]](sub, cfg, np.random.default_rng(seed))

    out = df.drop(columns=["is_poisoned"], errors="ignore").copy()
    for c in changed_cols:
        out.loc[train, c] = out_sub[c].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    mask[np.flatnonzero(train)] = m_sub
    return out, mask


def attack_from_config(df, attack_name, budget, timing, seed, dataset="paysim", config=None):
    """YAML-driven wrapper: looks up attack_name under `attacks.methods` in config/default.yaml."""
    config = config or load_config()
    acfg = config["attacks"]
    if dataset not in config["datasets"]:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(config['datasets'])}")
    if attack_name not in acfg["methods"]:
        raise ValueError(f"unknown attack {attack_name!r}; known: {sorted(acfg['methods'])}")
    cfg = dict(acfg["methods"][attack_name], label_col=acfg["label_col"], time_col=acfg["time_col"])
    if cfg["type"] == "feature_poison":
        cfg["features"] = list(acfg["features"][dataset])
    elif cfg["type"] == "backdoor":
        cfg["trigger"] = dict(acfg["trigger"][dataset])
    ds = config["datasets"][dataset]
    cw = acfg.get("clean_windows", 0)
    clean_until = None
    if cw and len(df):
        clean_until = int(df[acfg["time_col"]].min()) + cw * ds["window_steps"]
    return attack(df, budget, timing, seed, cfg, test_from_step=ds["test_from_step"], clean_until_step=clean_until)
