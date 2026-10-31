import numpy as np
from .common import feature_cols, pick, finish


def feature_poison(batch, cfg, rng):
    """cfg: rate, mode in {shift, scale, noise}, strength, features (optional),
    target_label (optional: poison only rows with this label).
    shift: x + strength*std | scale: x*strength | noise: x + N(0, (strength*std)^2)
    std is taken per feature from the incoming batch."""
    out = batch.copy()
    cols = feature_cols(out, cfg)
    cand = np.arange(len(out))
    if cfg.get("target_label") is not None:
        cand = np.flatnonzero(out["label"].to_numpy() == cfg["target_label"])
    idx = pick(cand, cfg["rate"], rng)
    if len(idx):
        X = out[cols].to_numpy(dtype=float, copy=True)
        sd = X.std(axis=0)
        s, mode = cfg["strength"], cfg["mode"]
        if mode == "shift":
            X[idx] += s * sd
        elif mode == "scale":
            X[idx] *= s
        elif mode == "noise":
            X[idx] += rng.normal(0.0, 1.0, X[idx].shape) * s * sd
        else:
            raise ValueError(f"unknown mode {mode}")
        out[cols] = X
    return finish(out, idx)
