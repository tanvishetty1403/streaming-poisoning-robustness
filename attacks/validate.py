import math

KNOWN_TYPES = {"random_label_flip", "targeted_label_flip", "feature_poison"}
MODES = {"shift", "scale", "noise"}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def validate(cfg: dict) -> dict:
    """Raise ValueError on a bad cfg. Runs even when rate == 0."""
    t = cfg.get("type")
    if t not in KNOWN_TYPES:
        raise ValueError(f"unknown attack type: {t!r}")
    rate = cfg.get("rate")
    if not _num(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be a number in [0, 1], got {rate!r}")
    if not isinstance(cfg.get("label_col", "label"), str):
        raise ValueError("label_col must be a string")
    if t == "feature_poison":
        mode = cfg.get("mode")
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode!r}")
        if not _num(cfg.get("strength")):
            raise ValueError("strength must be a number")
        feats = cfg.get("features")
        if not isinstance(feats, (list, tuple)) or not feats:
            raise ValueError("feature_poison requires a non-empty cfg['features']")
        if mode in ("shift", "noise"):
            fs = cfg.get("feature_std")
            if not isinstance(fs, dict) or any(f not in fs or not _num(fs[f]) or fs[f] < 0 for f in feats):
                raise ValueError("shift/noise need cfg['feature_std'] (>= 0) for every feature")
        if cfg.get("target_label") not in (None, 0, 1):
            raise ValueError("target_label must be None, 0 or 1")
    return cfg
