import numpy as np


def stamp_trigger(df, trigger: dict, idx=None):
    """Copy of df with the trigger values written into the trigger columns (rows idx, or all rows).
    Column dtypes are preserved (integer columns get the rounded value)."""
    out = df.copy()
    pos = np.arange(len(out)) if idx is None else np.asarray(idx, dtype=int)
    for c, v in trigger.items():
        if c not in out.columns:
            raise ValueError(f"trigger column not in df: {c!r}")
        col = out[c].to_numpy(copy=True)
        if col.dtype.kind not in "iuf":
            raise ValueError(f"trigger column {c!r} must be numeric, got {col.dtype}")
        if col.dtype.kind in "iu":
            info = np.iinfo(col.dtype)
            v = np.clip(np.rint(v), info.min, info.max)
        col[pos] = np.asarray(v).astype(col.dtype)
        out[c] = col
    return out
