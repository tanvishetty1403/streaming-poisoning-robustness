import numpy as np
import pandas as pd


class Quarantine:
    """Hold suspicious records, optionally releasing or discarding them after a review window.

    review_windows=None: quarantined records stay out for good (strict contract: every window
    row ends up in exactly one output). review_windows=N: after N windows each held record is
    re-scored and released if clean, else discarded. Every decision is appended to `log`
    (one dict per record per event), so a record's history can be reconstructed for the audit.
    """

    def __init__(self, review_windows=None):
        self.review_windows = review_windows
        self._held = []  # (due_window, records DataFrame, scores ndarray)
        self.log = []
        self.n_quarantined = 0

    def hold(self, records: pd.DataFrame, scores, window_id: int, reasons=None) -> None:
        if len(records) == 0:
            return
        reasons = reasons if reasons is not None else [None] * len(records)
        for rid, s, why in zip(records["record_id"], scores, reasons):
            self.log.append({"window_id": window_id, "record_id": rid, "event": "held",
                             "score": float(s), "detector": why})
        self.n_quarantined += len(records)
        if self.review_windows is not None:
            self._held.append((window_id + self.review_windows, records, np.asarray(scores)))

    def review(self, window_id: int, rescore, threshold: float):
        """Decide records whose review window has elapsed. Returns (released, discarded)."""
        keep, released, discarded = [], [], []
        template = self._held[0][1].iloc[0:0] if self._held else pd.DataFrame()   # keeps the schema
        for due, recs, sc in self._held:
            if due > window_id:
                keep.append((due, recs, sc))
                continue
            new = np.asarray(rescore(recs))
            ok = new <= threshold
            for rid, s, o in zip(recs["record_id"], new, ok):
                self.log.append({"window_id": window_id, "record_id": rid,
                                 "event": "released" if o else "discarded", "score": float(s),
                                 "detector": None})
            released.append(recs[ok])
            discarded.append(recs[~ok])
        self._held = keep
        return (pd.concat(released) if released else template,
                pd.concat(discarded) if discarded else template)

    @property
    def n_held(self) -> int:
        return sum(len(r) for _, r, _ in self._held)
