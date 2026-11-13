"""LightGBM baseline: window-level features, recording-level averaged probabilities."""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

from ml.audio import SR, AudioError, load_audio, windows
from ml.features import stat_features
from ml.metrics import icbhi_score, macc, report

log = logging.getLogger(__name__)
MIN_CYCLE_S = 2.0
LGBM = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, class_weight="balanced",
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)


def _segments(row, kind):
    """Windows for one row. Rows with start/end (ICBHI cycles) give one tiled window."""
    x = load_audio(row["path"], kind)
    if "start" in row and pd.notna(row.get("start")):
        seg = x[int(row["start"] * SR):int(row["end"] * SR)]
        if len(seg) < SR // 4:
            return []
        return [np.resize(seg, max(len(seg), int(MIN_CYCLE_S * SR)))]
    return windows(x, kind)


def _row_features(row, kind):
    try:
        return [stat_features(w, kind) for w in _segments(row, kind)]
    except AudioError as e:
        log.warning("skip %s: %s", row["path"], e)
        return []


def featurize(df, kind, n_jobs=-1):
    """Returns (X, y, groups, rec_ids, patients) at window level; rec_id indexes df rows."""
    feats = Parallel(n_jobs=n_jobs)(delayed(_row_features)(r, kind) for _, r in df.iterrows())
    X, rec = [], []
    for i, fs in enumerate(feats):
        X += fs
        rec += [i] * len(fs)
    rec = np.array(rec)
    return np.array(X), df["label"].to_numpy()[rec], rec, df["patient_id"].to_numpy()[rec]


def _lgbm(n_rows):
    return LGBMClassifier(**LGBM, min_child_samples=max(2, min(20, n_rows // 20)))


def _rec_proba(model, X, rec):
    p = pd.DataFrame(model.predict_proba(X)).groupby(rec).mean()
    return p.to_numpy(), p.index.to_numpy()


def _best_threshold(X, y, rec, pats, labels, positive):
    """Out-of-fold, patient-grouped choice of the decision threshold maximising MAcc."""
    oof_p, oof_y = [], []
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for tr, va in cv.split(X, y, pats):
        m = _lgbm(len(tr)).fit(X[tr], y[tr])
        p, idx = _rec_proba(m, X[va], rec[va])
        oof_p += list(p[:, list(m.classes_).index(positive)])
        oof_y += list(pd.Series(y[va]).groupby(rec[va]).first().loc[idx])
    oof_p, oof_y = np.array(oof_p), np.array(oof_y) == positive
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([macc(oof_y, oof_p >= t, positive=True) for t in grid])])


def train_task(df, kind, task_name, out_dir, positive=None, normal_label=None):
    df = df.reset_index(drop=True)
    tr, te = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    Xtr, ytr, rtr, ptr = featurize(tr, kind)
    Xte, _, rte, _ = featurize(te, kind)
    model = _lgbm(len(Xtr)).fit(Xtr, ytr)
    labels = list(model.classes_)
    threshold = _best_threshold(Xtr, ytr, rtr, ptr, labels, positive) if positive else None

    p, idx = _rec_proba(model, Xte, rte)
    y_true = list(te["label"].iloc[idx])
    if positive:
        neg = [l for l in labels if l != positive][0]
        y_pred = [positive if q >= threshold else neg for q in p[:, labels.index(positive)]]
    else:
        y_pred = [labels[i] for i in p.argmax(1)]

    r = report(y_true, y_pred, labels) | {"task": task_name, "kind": kind, "threshold": threshold,
                                           "n_train_rec": len(tr), "n_test_rec": len(te),
                                           "n_test_patients": te["patient_id"].nunique()}
    if positive:
        r["macc"] = macc([t == positive for t in y_true], [q == positive for q in y_pred], positive=True)
    if normal_label:
        r["icbhi"] = icbhi_score(y_true, y_pred, normal_label=normal_label)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": labels, "kind": kind, "threshold": threshold,
                 "positive": positive, "trained_on": ",".join(sorted(df["dataset"].unique()))},
                out / f"{task_name}.joblib", compress=3)
    log.info("%s: %s", task_name, json.dumps({k: v for k, v in r.items() if k != "confusion"}))
    return r
