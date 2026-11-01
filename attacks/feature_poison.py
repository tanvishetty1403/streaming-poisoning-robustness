import numpy as np
from .common import pick, finish


def feature_poison(batch, cfg, rng):
    """cfg: rate, mode in {shift, scale, noise}, strength, features (required),
    feature_std (required for shift/noise; from clean warm-up data), target_label (optional).
    shift: x + strength*ref_std | scale: x*strength | noise: x + N(0,1)*strength*ref_std
    Only rows whose values actually changed are marked poisoned."""
    out = batch.copy()
    cols = list(cfg["features"])
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"features not in batch: {missing}")
    cand = np.arange(len(out))
    if cfg.get("target_label") is not None:
        lc = cfg.get("label_col", "label")
        cand = np.flatnonzero(out[lc].to_numpy() == cfg["target_label"])
    idx = pick(cand, cfg["rate"], rng)
    if len(idx):
        X = out[cols].to_numpy(dtype=float, copy=True)
        old = X[idx].copy()
        s, mode = cfg["strength"], cfg["mode"]
        if mode == "scale":
            new = old * s
        else:
            sd = np.array([cfg["feature_std"][c] for c in cols], dtype=float)
            new = old + s * sd if mode == "shift" else old + rng.normal(size=old.shape) * s * sd
        X[idx] = new
        out[cols] = X
        same = (new == old) | (np.isnan(new) & np.isnan(old))
        idx = idx[~same.all(axis=1)]
    return finish(out, idx)
