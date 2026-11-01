import numpy as np
from sklearn.metrics import average_precision_score, recall_score, f1_score


def eval_clean(learner, X_test, y_test, threshold=0.5):
    """Per-window quality, measured on the clean held-out test set."""
    p = learner.predict_proba(X_test)
    pred = (p >= threshold).astype(int)
    if y_test.sum() == 0:
        return {"pr_auc": np.nan, "recall": np.nan, "f1": np.nan}
    return {
        "pr_auc": average_precision_score(y_test, p),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }


def eval_detection(window_df, quarantined_df):
    """Quarantine precision/recall against ground-truth poison labels."""
    truth = window_df["is_poisoned"].to_numpy()
    if truth.sum() == 0:
        return {"det_precision": np.nan, "det_recall": np.nan}
    flagged = window_df["record_id"].isin(quarantined_df["record_id"]).to_numpy()
    tp = (flagged & truth).sum()
    return {
        "det_precision": tp / flagged.sum() if flagged.sum() else 0.0,
        "det_recall": tp / truth.sum(),
    }