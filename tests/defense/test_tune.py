import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from tune_defense import run_file, subgroup_loss, summarize          # noqa: E402
from helpers import make_paysim, poison_amount                       # noqa: E402
from src.defense.pipeline import load_defense_cfg                    # noqa: E402


def stream(n_windows=4, poison_from=2):
    parts = []
    for k in range(n_windows):
        w = make_paysim(3000, seed=k, start=3000 * k, step0=1 + 24 * k)
        mask = np.zeros(len(w), dtype=bool)
        if k >= poison_from:
            w, mask = poison_amount(w, 0.1, 1e4, seed=k)
        parts.append(w.assign(is_poisoned=mask))
    return pd.concat(parts, ignore_index=True)


def test_run_file_scores_without_leaking_the_truth():
    df = stream()
    d, flagged, pw = run_file(df, load_defense_cfg(), 24)
    s = summarize(d, flagged, pw)
    assert s["recall"] >= 0.8 and s["clean_loss"] <= 0.02
    assert s["time_to_detect"] == 0.0                                  # caught in the first poisoned window
    assert pw["n_poisoned"].iloc[:2].sum() == 0                        # warm-up windows are clean
    assert {"type", "amount_bucket"} <= {k.split("=")[0].replace("loss_", "") for k in subgroup_loss(d, flagged)}
