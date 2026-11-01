import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


class OnlineLearner:
    """Logistic regression updated window by window (partial_fit)."""

    def __init__(self, fraud_weight=20.0, seed=0):
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=seed)
        self.fraud_weight = fraud_weight
        self.fitted = False

    def partial_fit(self, X, y):
        if len(X) == 0:
            return
        self.scaler.partial_fit(X)
        w = np.where(y == 1, self.fraud_weight, 1.0)   # fraud is ~0.1% of data
        self.clf.partial_fit(self.scaler.transform(X), y, classes=[0, 1], sample_weight=w)
        self.fitted = True

    def predict_proba(self, X):
        if not self.fitted:
            return np.zeros(len(X))
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]