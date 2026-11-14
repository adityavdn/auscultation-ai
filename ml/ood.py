"""Out-of-distribution gate: rejects audio whose features look nothing like the training recordings.

LightGBM is overconfident far from its training data (white noise came out as "COPD 100%"), so the
probability threshold alone can't catch it.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

FALSE_REJECT_RATE = 0.002  # threshold at the 0.2th percentile of training-window scores (1% rejected short Yaseen clips)


def fit_ood(X):
    iso = IsolationForest(n_estimators=200, random_state=42).fit(X)
    return {"iso": iso, "threshold": float(np.quantile(iso.score_samples(X), FALSE_REJECT_RATE))}


def is_ood(bundle, X):
    """True when the median window of a recording scores below the training threshold."""
    return bool(np.median(bundle["iso"].score_samples(X)) < bundle["threshold"])


def train_ood(frames, kind, out_dir):
    """Fit the gate on train-split windows of every dataset for this kind; returns the window count."""
    from pathlib import Path

    import joblib
    import pandas as pd

    from ml.train_baseline import featurize
    df = pd.concat([f[f.split == "train"] for f in frames], ignore_index=True)
    df = df.drop(columns=[c for c in ("start", "end") if c in df]).drop_duplicates("path")
    X, *_ = featurize(df, kind)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(fit_ood(X), Path(out_dir) / f"ood_{kind}.joblib", compress=3)
    return len(X)
