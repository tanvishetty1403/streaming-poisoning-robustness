import numpy as np
import pandas as pd

PAYSIM = "data/raw/paysim.csv"
ULB = "data/raw/creditcard.csv"
LABEL = "isFraud"
TIME = "step"


def load_dataset(name, cfg, seed=0):
    """Return a raw dataframe in event-time order with columns: record_id, step, isFraud, ...
    Every dataset is normalised to use 'step' (event time) and 'isFraud' (label)."""
    if name == "paysim":
        df = pd.read_csv(PAYSIM)
    elif name == "ulb":
        df = pd.read_csv(ULB).rename(columns={"Time": TIME, "Class": LABEL})
        df[TIME] = (df[TIME] // 3600).astype(int)          # seconds -> hours
    else:
        raise ValueError(f"unknown dataset {name}")
    frac = cfg["sample_frac"]
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=seed)
    df = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    df.insert(0, "record_id", np.arange(len(df)))
    return df


def split_stream_test(df, test_from_step):
    """Time-based split: the stream is the past, the clean test set is the future (never attacked)."""
    stream = df[df[TIME] < test_from_step].reset_index(drop=True)
    test = df[df[TIME] >= test_from_step].reset_index(drop=True)
    return stream, test


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))      # tames heavy-tailed money amounts


def featurize(df, name):
    """Raw columns -> model matrix. Called AFTER any attack, so poisoned raw
    values flow through the same feature code as clean ones."""
    if name == "paysim":
        X = pd.DataFrame({
            "amount": _slog(df["amount"]),
            "oldO": _slog(df["oldbalanceOrg"]), "newO": _slog(df["newbalanceOrig"]),
            "oldD": _slog(df["oldbalanceDest"]), "newD": _slog(df["newbalanceDest"]),
            "errO": _slog(df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]),
            "errD": _slog(df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]),
        })
        for t in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]:
            X[f"type_{t}"] = (df["type"] == t).astype(float)
        return X.to_numpy(dtype=float)
    cols = [f"V{i}" for i in range(1, 29)]
    X = df[cols].copy()
    X["Amount"] = _slog(df["Amount"])
    return X.to_numpy(dtype=float)