"""Evaluation metrics used in auscultation challenges (ICBHI 2017, PhysioNet/CinC 2016)."""
from sklearn.metrics import confusion_matrix, f1_score


def icbhi_score(y_true, y_pred, normal_label="normal"):
    """Official ICBHI score: Se = exact matches on abnormal cases, Sp = correct normals."""
    pairs = list(zip(y_true, y_pred))
    normals = [(t, p) for t, p in pairs if t == normal_label]
    abnormals = [(t, p) for t, p in pairs if t != normal_label]
    sp = sum(t == p for t, p in normals) / len(normals) if normals else 0.0
    se = sum(t == p for t, p in abnormals) / len(abnormals) if abnormals else 0.0
    return {"se": se, "sp": sp, "score": (se + sp) / 2}


def sensitivity_specificity(y_true, y_pred, positive_labels):
    pos = set(positive_labels)
    tp = sum(t in pos and p in pos for t, p in zip(y_true, y_pred))
    tn = sum(t not in pos and p not in pos for t, p in zip(y_true, y_pred))
    n_pos = sum(t in pos for t in y_true)
    n_neg = len(y_true) - n_pos
    return (tp / n_pos if n_pos else 0.0, tn / n_neg if n_neg else 0.0)


def macc(y_true, y_pred, positive=1):
    """PhysioNet 2016 MAcc: mean of sensitivity and specificity for a binary task."""
    se, sp = sensitivity_specificity(y_true, y_pred, [positive])
    return (se + sp) / 2


def report(y_true, y_pred, labels):
    return {
        "n": len(y_true),
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "labels": list(labels),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
