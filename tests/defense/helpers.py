"""Independent dev helpers so the defense can be built and tested without P2's attacks."""
import numpy as np
import pandas as pd


def make_clean(n=1000, d=3, seed=0, start=0):
    """Features are standard normal; labels are random (independent of the features)."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame(r.normal(size=(n, d)), columns=[f"V{i + 1}" for i in range(d)])
    df.insert(0, "record_id", np.arange(start, start + n))
    df.insert(1, "event_time", np.arange(start, start + n, dtype=float))
    df["label"] = (r.random(n) < 0.05).astype(int)
    return df


def make_labeled(n=1000, seed=0, start=0):
    """Like make_clean, but the label is a function of the features (so a model can learn it)."""
    df = make_clean(n, seed=seed, start=start)
    df["label"] = ((df["V1"] + 0.5 * df["V2"]) > 1.0).astype(int)
    return df


def shift_rows(df, frac, mult, seed=0):
    """Shift all features of a random fraction of rows by `mult` std. Returns (df, positional bool mask)."""
    r = np.random.default_rng(seed)
    out = df.copy()
    mask = np.zeros(len(out), dtype=bool)
    mask[r.choice(len(out), size=int(round(frac * len(out))), replace=False)] = True
    cols = [c for c in out.columns if c.startswith("V")]
    out.loc[out.index[mask], cols] = out.loc[out.index[mask], cols].to_numpy() + mult
    return out, mask


def flip_labels(df, frac, seed=0, exclude=None):
    """Flip the label of a random fraction of rows (optionally not those in `exclude`)."""
    r = np.random.default_rng(seed)
    out = df.copy()
    pool = np.arange(len(out)) if exclude is None else np.flatnonzero(~exclude)
    mask = np.zeros(len(out), dtype=bool)
    mask[r.choice(pool, size=int(round(frac * len(out))), replace=False)] = True
    out.loc[out.index[mask], "label"] = 1 - out.loc[out.index[mask], "label"].to_numpy()
    return out, mask


# ---------------- PaySim-shaped raw windows (for testing defend() with P1's real featurizer) ----------------
def make_paysim(n=4000, seed=0, start=0, step0=1, steps=24):
    """Raw PaySim-like window: balances obey the usual identities; ~2% of TRANSFER/CASH_OUT rows are
    'drain' frauds (amount == old balance, new balance 0), so a model can learn the label."""
    r = np.random.default_rng(seed)
    types = r.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n, p=[.35, .08, .35, .21, .01])
    amt = np.round(np.exp(r.normal(9, 1.2, n)), 2)
    old_o = np.round(amt * (1 + np.exp(r.normal(0.5, 1.0, n))), 2)
    new_o = np.where(types == "CASH_IN", old_o + amt, old_o - amt)
    old_d = np.where(types == "PAYMENT", 0.0, np.round(np.exp(r.normal(11, 1.5, n)), 2))
    new_d = np.where(types == "PAYMENT", 0.0, np.where(types == "CASH_IN", old_d, old_d + amt))
    fraud = np.zeros(n, dtype=int)
    cand = np.flatnonzero(np.isin(types, ["TRANSFER", "CASH_OUT"]))
    f = r.choice(cand, size=max(1, int(0.02 * len(cand))), replace=False)
    amt[f], new_o[f], fraud[f] = old_o[f], 0.0, 1
    df = pd.DataFrame({
        "step": r.integers(step0, step0 + steps, n), "type": types, "amount": amt,
        "nameOrig": [f"C{i}" for i in range(n)], "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
        "nameDest": [f"M{i}" if t == "PAYMENT" else f"C{i}" for i, t in enumerate(types)],
        "oldbalanceDest": old_d, "newbalanceDest": new_d, "isFraud": fraud, "isFlaggedFraud": 0})
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    df.insert(0, "record_id", np.arange(start, start + n))
    return df


def poison_amount(df, frac, mult, seed=0):
    """Multiply `amount` of a random fraction of rows. Returns (df, positional bool mask)."""
    r = np.random.default_rng(seed)
    out = df.copy()
    mask = np.zeros(len(out), dtype=bool)
    mask[r.choice(len(out), size=int(round(frac * len(out))), replace=False)] = True
    out.loc[out.index[mask], "amount"] = out.loc[out.index[mask], "amount"].to_numpy() * mult
    return out, mask


def flip_fraud_labels(df, frac, seed=0):
    r = np.random.default_rng(seed)
    out = df.copy()
    mask = np.zeros(len(out), dtype=bool)
    mask[r.choice(len(out), size=int(round(frac * len(out))), replace=False)] = True
    out.loc[out.index[mask], "isFraud"] = 1 - out.loc[out.index[mask], "isFraud"].to_numpy()
    return out, mask
