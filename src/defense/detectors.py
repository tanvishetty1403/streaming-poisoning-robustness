"""Record-level detectors. Common interface:

    name, threshold, ready
    update(X, y)  -> learn from ACCEPTED records only (never from quarantined ones)
    score(X, y)   -> one raw suspicion score per record; a record is suspicious above `threshold`
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _median_mad(ref):
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0)
    return med, 1.4826 * mad, mad > 0


class RobustZ:
    """Median/MAD robust z-score against a rolling reference of ACCEPTED records.

    score = max over features of |x - median| / (1.4826 * MAD). It catches feature poisoning
    (shift / scale / noise). Features with MAD == 0 in a reference (constant or mostly-constant
    columns) are ignored: they need a distribution detector.

    class_conditional=False (marginal): every record is compared with ALL accepted records.
        Problem: fraud is rare and separable, so true fraud looks like an outlier and gets
        quarantined, starving the learner of the minority class.
    class_conditional=True: a record is compared with accepted records that carry the SAME
        label, so true fraud is judged against fraud. A fraud-labelled row that looks legit
        (a label flip) or a row shifted away from its class is an outlier. A class stays silent
        until it has `min_ref_class` accepted records.
    Marginal mode only: with `bootstrap`, a window scored before any reference exists uses its
    own median/MAD (robust while < 50% of it is poisoned).
    """
    name = "robust_z"

    def __init__(self, threshold, window, min_ref, bootstrap=False, class_conditional=False, min_ref_class=30):
        self.threshold, self.window, self.min_ref = threshold, window, min_ref
        self.bootstrap, self.class_conditional, self.min_ref_class = bootstrap, class_conditional, min_ref_class
        self.needs_labels = class_conditional
        self._ref = {}    # key (None, or the class 0/1) -> accepted rows
        self._stats = {}  # key -> (median, scale, active)

    def _add(self, key, X):
        ref = X if key not in self._ref else np.vstack([self._ref[key], X])
        self._ref[key] = ref[-self.window:]
        self._stats[key] = _median_mad(self._ref[key])

    def _min(self) -> int:
        return self.min_ref_class if self.class_conditional else self.min_ref

    @property
    def ready(self) -> bool:
        return any(len(r) >= self._min() for r in self._ref.values())

    def update(self, X, y=None) -> None:
        if len(X) == 0:
            return
        if self.class_conditional:
            for c in (0, 1):
                Xc = X[y == c]
                if len(Xc):
                    self._add(c, Xc)
        else:
            self._add(None, X)

    @staticmethod
    def _z(X, stats):
        med, scale, active = stats
        if not active.any():
            return np.zeros(len(X))
        z = np.zeros(X.shape, dtype=float)
        z[:, active] = np.abs(X[:, active] - med[active]) / scale[active]
        return z.max(axis=1)

    def score(self, X, y=None) -> np.ndarray:
        out = np.zeros(len(X))
        if len(X) == 0:
            return out
        if self.class_conditional:
            for c in (0, 1):
                m = y == c
                if m.any() and c in self._ref and len(self._ref[c]) >= self.min_ref_class:
                    out[m] = self._z(X[m], self._stats[c])
            return out
        if None in self._ref and len(self._ref[None]) >= self.min_ref:
            return self._z(X, self._stats[None])
        if self.bootstrap and len(X) >= self.min_ref:
            return self._z(X, _median_mad(X))
        return out


class LabelConsistency:
    """Loss of each record's label under a reference model trained on ACCEPTED records.

    score = -log p(label | x). A record whose label the reference model confidently
    contradicts (a fraud-looking row labelled legit, or the reverse) scores high, which is
    the signature of a label flip. The model is class-balanced (fraud is rare) and refit on a
    rolling window each time accepted records arrive. It stays silent (score 0) until the
    window holds `min_ref` records with at least `min_class` of each class.
    Limitation: flips of records near the decision boundary look normal and are not caught.
    """
    name = "label_loss"
    needs_labels = True

    def __init__(self, threshold, window, min_ref, min_class):
        self.threshold, self.window, self.min_ref, self.min_class = threshold, window, min_ref, min_class
        self._X = None
        self._y = None
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def update(self, X, y) -> None:
        if len(X) == 0:
            return
        self._X = X if self._X is None else np.vstack([self._X, X])
        self._y = y if self._y is None else np.concatenate([self._y, y])
        self._X, self._y = self._X[-self.window:], self._y[-self.window:]
        n1 = int((self._y == 1).sum())
        n0 = len(self._y) - n1
        if len(self._y) >= self.min_ref and min(n0, n1) >= self.min_class:
            self._model = make_pipeline(
                StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=200)
            ).fit(self._X, self._y)

    def score(self, X, y) -> np.ndarray:
        if self._model is None or len(X) == 0:
            return np.zeros(len(X))
        p1 = self._model.predict_proba(X)[:, 1]
        p = np.where(y == 1, p1, 1.0 - p1)
        return -np.log(np.clip(p, 1e-6, 1.0))
