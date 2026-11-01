"""Detection metrics. Evaluator-side only: poison_mask is ground truth and never goes to the defense."""
import numpy as np


def detection_metrics(flagged, poison_mask) -> dict:
    """Per-record precision, recall, F1 and false-positive rate. Both inputs are aligned bool arrays."""
    f = np.asarray(flagged, dtype=bool)
    p = np.asarray(poison_mask, dtype=bool)
    tp, fp = int((f & p).sum()), int((f & ~p).sum())
    fn, tn = int((~f & p).sum()), int((~f & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
