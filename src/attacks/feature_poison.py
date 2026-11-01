import numpy as np
from .common import pick, finish


def feature_poison(batch, cfg, rng):
    """cfg: rate, timing, mode in {shift, scale, noise}, strength, features (required, numeric),
    feature_std (required for shift/noise; reference std), target_label (None/0/1), label_col.
    shift: x + strength*ref_std | scale: x*strength | noise: x + N(0,1)*strength*ref_std
    Column dtypes are preserved (integer columns are rounded back to int).
    Only rows whose values actually changed are marked poisoned."""
    lc = cfg.get("label_col", "isFraud")
    out = batch.copy()
    cols = list(cfg["features"])
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise ValueError(f"features not in batch: {missing}")
    cand = np.arange(len(out))
    if cfg.get("target_label") is not None:
        cand = np.flatnonzero(out[lc].to_numpy() == cfg["target_label"])
    idx = pick(cand, cfg["rate"], rng, cfg.get("timing", "uniform"))
    if len(idx) == 0:
        return finish(out, idx)
    s, mode = cfg["strength"], cfg["mode"]
    changed = np.zeros(len(idx), dtype=bool)
    for c in cols:
        col = out[c].to_numpy(copy=True)
        if col.dtype.kind not in "iuf":
            raise ValueError(f"feature {c!r} must be numeric, got {col.dtype}")
        old = col[idx]
        oldf = old.astype(float)
        if mode == "scale":
            new = oldf * s
        else:
            sd = float(cfg["feature_std"][c])
            new = oldf + s * sd if mode == "shift" else oldf + rng.normal(size=len(idx)) * s * sd
        if col.dtype.kind in "iu":
            info = np.iinfo(col.dtype)
            new = np.clip(np.rint(new), info.min, info.max)
        newc = new.astype(col.dtype)
        same = (newc == old) | (np.isnan(newc.astype(float)) & np.isnan(oldf))
        changed |= ~same
        col[idx] = newc
        out[c] = col
    return finish(out, idx[changed])
