"""Dataset loaders. Each returns a DataFrame: path, patient_id, label, dataset, split.

Splits are patient-wise: no patient appears in both train and test.
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
YASEEN = {"N": "normal", "AS": "AS", "MS": "MS", "MR": "MR", "MVP": "MVP"}
# Diagnoses grouped into clinically coherent classes: several raw labels have only 1-6 patients,
# too few to learn or to evaluate on their own.
DIAG_MAP = {
    "healthy": "healthy", "n": "healthy", "normal": "healthy",
    "copd": "copd", "asthma": "asthma", "heart failure": "heart_failure",
    "pneumonia": "infection", "urti": "infection", "lrti": "infection", "bronchiolitis": "infection",
    "bronchiectasis": "other",
}


def patient_split(df, n_splits=5, seed=SEED):
    """Fold 0 of a stratified, patient-grouped K-fold becomes the test set (~20%)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_idx = next(cv.split(df, df["label"], groups=df["patient_id"]))
    split = pd.Series("train", index=df.index)
    split.iloc[test_idx] = "test"
    return df.assign(split=split.values)


def load_cinc2016(root):
    # ponytail: CinC 2016 publishes no patient IDs, so each record is its own group.
    rows = []
    for ref in sorted(Path(root).glob("training-*/REFERENCE.csv")):
        for line in ref.read_text().split():
            rec, lab = line.split(",")
            rows.append({"path": str(ref.parent / f"{rec}.wav"), "patient_id": f"cinc_{rec}",
                         "label": "abnormal" if lab.strip() == "1" else "normal"})
    return patient_split(pd.DataFrame(rows).assign(dataset="cinc2016"))


def load_circor(root):
    root = Path(root)
    meta = pd.read_csv(root / "training_data.csv")
    meta = meta[meta["Murmur"].isin(["Present", "Absent"])]
    rows = [{"path": str(w), "patient_id": f"circor_{r['Patient ID']}", "label": r["Murmur"].lower()}
            for _, r in meta.iterrows()
            for w in sorted((root / "training_data").glob(f"{r['Patient ID']}_*.wav"))]
    return patient_split(pd.DataFrame(rows).assign(dataset="circor"))


def load_yaseen(root):
    rows = []
    for w in sorted(Path(root).rglob("*.wav")):
        key = w.parent.name.split("_")[0]
        if key in YASEEN:  # ponytail: no patient IDs published; each file is its own group.
            rows.append({"path": str(w), "patient_id": f"yaseen_{w.stem}", "label": YASEEN[key]})
    return patient_split(pd.DataFrame(rows).assign(dataset="yaseen"))


def _cycle_label(crackle, wheeze):
    return {(0, 0): "normal", (1, 0): "crackle", (0, 1): "wheeze", (1, 1): "both"}[(crackle, wheeze)]


def load_icbhi_cycles(root, split_file):
    """One row per annotated respiratory cycle, using the official 60/40 patient split."""
    official = dict(l.split() for l in Path(split_file).read_text().splitlines() if l.strip())
    rows = []
    for txt in sorted(Path(root).glob("*.txt")):
        if txt.stem not in official:
            continue
        for line in txt.read_text().splitlines():
            if not line.strip():
                continue
            start, end, c, w = line.split()
            rows.append({"path": str(txt.with_suffix(".wav")), "patient_id": f"icbhi_{txt.stem.split('_')[0]}",
                         "label": _cycle_label(int(c), int(w)), "start": float(start), "end": float(end),
                         "split": official[txt.stem]})
    return pd.DataFrame(rows).assign(dataset="icbhi")


def _kauh_row(w):
    m = re.match(r"^[A-Z]P(\d+)_([^,]+),", w.name)
    if not m:
        return None
    diag = m.group(2).strip().lower()
    return {"path": str(w), "patient_id": f"kauh_{m.group(1)}", "label": DIAG_MAP.get(diag, "other"),
            "dataset": "kauh"}


def load_lung_diagnosis(icbhi_root, diag_file, kauh_root=None):
    """One row per recording, labelled with the patient's diagnosis."""
    diag = dict(l.split("\t") for l in Path(diag_file).read_text().splitlines() if l.strip())
    rows = [{"path": str(w), "patient_id": f"icbhi_{w.stem.split('_')[0]}",
             "label": DIAG_MAP.get(diag[w.stem.split("_")[0]].strip().lower(), "other"), "dataset": "icbhi"}
            for w in sorted(Path(icbhi_root).glob("*.wav")) if w.stem.split("_")[0] in diag]
    if kauh_root and Path(kauh_root).exists():
        rows += [r for r in map(_kauh_row, sorted(Path(kauh_root).rglob("*.wav"))) if r]
    return patient_split(pd.DataFrame(rows))
