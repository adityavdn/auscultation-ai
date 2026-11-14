"""Out-of-distribution gate: rejects audio whose features look nothing like the training recordings.

LightGBM is overconfident far from its training data (white noise came out as "COPD 100%"), so the
probability threshold alone can't catch it.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

FALSE_REJECT_RATE = 0.01  # threshold sits at the 1st percentile of training-window scores


def fit_ood(X):
    iso = IsolationForest(n_estimators=200, random_state=42).fit(X)
    return {"iso": iso, "threshold": float(np.quantile(iso.score_samples(X), FALSE_REJECT_RATE))}


def is_ood(bundle, X):
    """True when the median window of a recording scores below the training threshold."""
    return bool(np.median(bundle["iso"].score_samples(X)) < bundle["threshold"])
