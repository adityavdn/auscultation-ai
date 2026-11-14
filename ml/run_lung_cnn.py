"""Fine-tune the lung-sound CNN on ICBHI breath cycles (official split) and export ONNX.

Two stages in separate processes (mixing joblib/librosa/LightGBM and torch in one process segfaulted):
  python -m ml.run_lung_cnn prep  <data_dir> <out_dir>                 # spectrograms -> out/mels.npz (no torch)
  python -m ml.run_lung_cnn train <out_dir> [epochs] [device]          # fine-tune + ONNX (no LightGBM)
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit

from ml.audio import SR, AudioError, load_audio
from ml.cnn_frontend import N_FRAMES, mel32k
from ml.metrics import icbhi_score, report

log = logging.getLogger("run_lung_cnn")
LABELS = ["both", "crackle", "normal", "wheeze"]
MIN_CYCLE = SR // 4


def _recording_mels(path, rows):
    try:
        x = load_audio(path, "lung")
    except AudioError:
        return []
    out = []
    for i, start, end in rows:
        seg = x[int(start * SR):int(end * SR)]
        if len(seg) >= MIN_CYCLE:
            out.append((i, mel32k(seg).astype(np.float16)))  # float16 halves memory on an 8 GB machine
    return out


def cycle_mels(df):
    jobs = [(p, list(zip(g.index, g.start, g.end))) for p, g in df.groupby("path")]
    res = Parallel(n_jobs=4)(delayed(_recording_mels)(p, r) for p, r in jobs)
    pairs = sorted(x for r in res for x in r)
    idx = [i for i, _ in pairs]
    return np.stack([m for _, m in pairs]), df.loc[idx, "label"].to_numpy(), df.loc[idx, "patient_id"].to_numpy()


def prep(data, out):
    from ml import datasets as D
    from ml.run_all import OPERA, _get, fetch_icbhi
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = D.load_icbhi_cycles(fetch_icbhi(data), _get(OPERA + "ICBHI_challenge_train_test.txt",
                                                    Path(data) / "icbhi_split.txt")).reset_index(drop=True)
    Xtr, ytr, ptr = cycle_mels(df[df.split == "train"])
    Xte, yte, pte = cycle_mels(df[df.split == "test"])
    np.savez(out / "mels.npz", Xtr=Xtr, ytr=ytr, ptr=ptr, Xte=Xte, yte=yte, pte=pte)
    log.info("saved %d train / %d test cycles", len(Xtr), len(Xte))


def train(out, epochs=12, device="cpu"):
    from ml.lung_cnn import _predict, export_onnx, fit
    out = Path(out)
    d = np.load(out / "mels.npz", allow_pickle=True)
    Xtr, ytr, ptr, Xte, yte, pte = (d[k] for k in ("Xtr", "ytr", "ptr", "Xte", "yte", "pte"))
    fit_i, val_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=42).split(Xtr, ytr, ptr))
    log.info("train %d / val %d / test %d cycles", len(fit_i), len(val_i), len(Xte))
    model, info = fit(Xtr[fit_i], ytr[fit_i], Xtr[val_i], ytr[val_i], LABELS, epochs=epochs, device=device)
    pred = [LABELS[k] for k in _predict(model, Xte, "cpu").argmax(1)]
    r = report(list(yte), pred, LABELS) | {"icbhi": icbhi_score(list(yte), pred), "task": "lung_sound",
                                            "model": "efficientnet_b0 (ImageNet) fine-tuned", **info,
                                            "n_test_cycles": len(Xte), "n_test_patients": len(set(pte))}
    export_onnx(model, out / "lung_sound.onnx", N_FRAMES)
    (out / "lung_sound.onnx.json").write_text(json.dumps({"labels": LABELS, "kind": "lung", "normal": "normal",
                                                          "trained_on": "icbhi"}))
    (out / "lung_sound_cnn_metrics.json").write_text(json.dumps(r, indent=2, default=float))
    log.info("TEST ICBHI %s", r["icbhi"])
    return r


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3])
    else:
        train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12, sys.argv[4] if len(sys.argv) > 4 else "cpu")
