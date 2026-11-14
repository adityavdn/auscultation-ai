"""Lung sounds with AudioSet-pretrained AST embeddings and a light head (ICBHI official split).

Three stages, each in its own process (torch next to joblib/LightGBM segfaulted on this machine):
  python -m ml.run_lung_ast prep  <data_dir> <out_dir>   # breath cycles at 4 kHz -> out/cycles.npz (no torch)
  python -m ml.run_lung_ast embed <out_dir> [device]     # frozen AST embeddings -> out/emb.npz (torch only)
  python -m ml.run_lung_ast head  <out_dir>              # choose head on train patients, score test once
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("run_lung_ast")
LABELS = ["both", "crackle", "normal", "wheeze"]
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000
CYCLE_S = 8.0  # each cycle is repeat-padded to 8 s, as in the Patch-Mix ICBHI recipe
LAYERS = (4, 8, 12)  # mean-pooled tokens from these hidden layers


def fit_cycle(seg, sr):
    """Repeat-pad (or cut) one breath cycle to CYCLE_S seconds."""
    return np.resize(np.asarray(seg, dtype=np.float32), int(CYCLE_S * sr))


# ---------- stage 1: cycles (no torch) ----------
def _recording_cycles(path, rows):
    from ml.audio import SR, AudioError, load_audio
    try:
        x = load_audio(path, "lung")
    except AudioError:
        return []
    out = []
    for i, start, end in rows:
        seg = x[int(start * SR):int(end * SR)]
        if len(seg) >= SR // 4:
            out.append((i, fit_cycle(seg, SR).astype(np.float16)))
    return out


def prep(data, out):
    from joblib import Parallel, delayed
    from ml import datasets as D
    from ml.run_all import OPERA, _get, fetch_icbhi
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = D.load_icbhi_cycles(fetch_icbhi(data), _get(OPERA + "ICBHI_challenge_train_test.txt",
                                                    Path(data) / "icbhi_split.txt")).reset_index(drop=True)
    parts = {}
    for split in ("train", "test"):
        sub = df[df.split == split]
        jobs = [(p, list(zip(g.index, g.start, g.end))) for p, g in sub.groupby("path")]
        res = Parallel(n_jobs=4)(delayed(_recording_cycles)(p, r) for p, r in jobs)
        pairs = sorted(x for r in res for x in r)
        idx = [i for i, _ in pairs]
        parts[split] = (np.stack([c for _, c in pairs]), df.loc[idx, "label"].to_numpy(), df.loc[idx, "patient_id"].to_numpy())
    np.savez(out / "cycles.npz", **{f"{k}{s}": v for s, (X, y, p) in parts.items() for k, v in zip("Xyp", (X, y, p))})
    log.info("saved %d train / %d test cycles", len(parts["train"][0]), len(parts["test"][0]))


# ---------- stage 2: embeddings (torch only) ----------
def load_ast(device):
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    fe = ASTFeatureExtractor.from_pretrained(AST_ID)
    model = ASTModel.from_pretrained(AST_ID, output_hidden_states=True).to(device).eval()
    torch.set_grad_enabled(False)
    return fe, model


def embed_batch(fe, model, waves_4k, device):
    """waves_4k: list of float arrays at 4 kHz -> (n, 768 * len(LAYERS)) embeddings."""
    import torch
    from scipy.signal import resample_poly
    waves = [resample_poly(w.astype(np.float32), AST_SR // 4000, 1) for w in waves_4k]
    feats = fe(waves, sampling_rate=AST_SR, return_tensors="pt")["input_values"].to(device)
    hs = model(feats).hidden_states
    return torch.cat([hs[i][:, 2:].mean(1) for i in LAYERS], 1).float().cpu().numpy()


def embed(out, device="cpu", batch=16):
    out = Path(out)
    d = np.load(out / "cycles.npz", allow_pickle=True)
    fe, model = load_ast(device)
    res = {}
    for split in ("train", "test"):
        X = d[f"X{split}"]
        chunks = []
        for i in range(0, len(X), batch):
            chunks.append(embed_batch(fe, model, list(X[i:i + batch]), device))
            if (i // batch) % 40 == 0:
                log.info("%s %d/%d", split, i, len(X))
        res[f"E{split}"] = np.concatenate(chunks).astype(np.float32)
    np.savez(out / "emb.npz", **res)
    log.info("embeddings %s", {k: v.shape for k, v in res.items()})


# ---------- stage 3: head (no torch) ----------
def _heads():
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")),
        "lgbm": lambda: LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, class_weight="balanced",
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=42, verbose=-1),
    }


def head(out):
    from sklearn.model_selection import GroupShuffleSplit
    from ml.metrics import icbhi_score, report
    out = Path(out)
    d, e = np.load(out / "cycles.npz", allow_pickle=True), np.load(out / "emb.npz")
    ytr, ptr, yte, pte = d["ytrain"], d["ptrain"], d["ytest"], d["ptest"]
    Etr, Ete = e["Etrain"], e["Etest"]
    fit_i, val_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(Etr, ytr, ptr))
    scores = {}
    for name, make in _heads().items():
        m = make().fit(Etr[fit_i], ytr[fit_i])
        scores[name] = icbhi_score(list(ytr[val_i]), list(m.predict(Etr[val_i])))["score"]
    best = max(scores, key=scores.get)
    log.info("validation (train patients only): %s -> %s", scores, best)
    model = _heads()[best]().fit(Etr, ytr)  # refit on all training patients, then touch the test set once
    pred = list(model.predict(Ete))
    r = report(list(yte), pred, LABELS) | {"icbhi": icbhi_score(list(yte), pred), "task": "lung_sound",
                                            "model": f"AST ({AST_ID}) frozen embeddings + {best}", "val_scores": scores,
                                            "n_test_cycles": len(yte), "n_test_patients": len(set(pte))}
    (out / "lung_sound_ast_metrics.json").write_text(json.dumps(r, indent=2, default=float))
    import joblib
    joblib.dump({"model": model, "labels": list(model.classes_), "kind": "lung", "threshold": None, "positive": None,
                 "trained_on": "icbhi", "embedding": AST_ID, "layers": list(LAYERS)}, out / "lung_sound_ast_head.joblib")
    log.info("TEST ICBHI %s", r["icbhi"])
    return r


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    stage, args = sys.argv[1], sys.argv[2:]
    if stage == "prep":
        prep(*args)
    elif stage == "embed":
        embed(args[0], args[1] if len(args) > 1 else "cpu")
    else:
        head(args[0])
