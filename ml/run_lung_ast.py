"""Lung sounds with a slimmed AudioSet-pretrained AST (ICBHI official split), built to run fast on a CPU.

The full frozen AST scored 0.533 but cost 2.3 s per window. This version keeps the first 8 of 12 layers and
reads 4 s of audio instead of 10.24 s (positional embeddings cropped to match), is quantised to int8, and its
classifier head is trained on the int8 embeddings themselves, so serving sees exactly what training saw.

Four stages, each in its own process (torch next to joblib/LightGBM segfaulted on this machine):
  python -m ml.run_lung_ast prep   <data_dir> <out_dir>  # breath cycles at 4 kHz -> out/cycles.npz (no torch)
  python -m ml.run_lung_ast export <out_dir>             # slim AST -> out/lung_sound_ast.onnx, int8 (torch only)
  python -m ml.run_lung_ast embed  <out_dir>             # int8 embeddings -> out/emb.npz (onnxruntime only)
  python -m ml.run_lung_ast head   <out_dir>             # choose head on train patients, score test once
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("run_lung_ast")
LABELS = ["both", "crackle", "normal", "wheeze"]
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR = 16000
CYCLE_S = 4.0  # each cycle is repeat-padded (or cut) to 4 s
MAX_FRAMES = 400  # fbank frames for 4 s at a 10 ms hop
N_LAYERS = 8  # transformer layers kept out of 12
LAYERS = (4, 8)  # mean-pooled tokens from these layers form the embedding


def fit_cycle(seg, sr):
    """Repeat-pad (or cut) one breath cycle to CYCLE_S seconds."""
    return np.resize(np.asarray(seg, dtype=np.float32), int(CYCLE_S * sr))


def to_fbank(waves_4k):
    """4 kHz cycles -> AST input frames (n, MAX_FRAMES, 128)."""
    from scipy.signal import resample_poly
    from ml.ast_fbank import fbanks
    return fbanks([resample_poly(fit_cycle(w, 4000), AST_SR // 4000, 1) for w in waves_4k], MAX_FRAMES)


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


# ---------- stage 2: slim AST -> int8 ONNX (torch only) ----------
def slim_ast():
    """The pretrained AST cut to N_LAYERS layers and MAX_FRAMES frames (positional embeddings cropped in time)."""
    import copy
    import torch
    from transformers import ASTModel
    full = ASTModel.from_pretrained(AST_ID)
    cfg = copy.deepcopy(full.config)
    cfg.max_length, cfg.num_hidden_layers, cfg.output_hidden_states = MAX_FRAMES, N_LAYERS, True
    small = ASTModel(cfg).eval()
    sd = full.state_dict()
    f = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
    t_full = (full.config.max_length - cfg.patch_size) // cfg.time_stride + 1
    t = (MAX_FRAMES - cfg.patch_size) // cfg.time_stride + 1
    pos = sd["embeddings.position_embeddings"]
    sd["embeddings.position_embeddings"] = torch.cat(
        [pos[:, :2], pos[:, 2:].reshape(1, f, t_full, -1)[:, :, :t].reshape(1, f * t, -1)], 1)
    keep = small.state_dict().keys()
    small.load_state_dict({k: v for k, v in sd.items() if k in keep}, strict=True)
    return small


def export(out):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out = Path(out)
    model = slim_ast()

    class Emb(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            hs = self.m(x).hidden_states
            return torch.cat([hs[i][:, 2:].mean(1) for i in LAYERS], 1)

    fp32 = out / "ast_slim_fp32.onnx"
    torch.onnx.export(Emb(model), (torch.zeros(2, MAX_FRAMES, 128),), str(fp32), input_names=["fbank"],
                      output_names=["emb"], dynamic_axes={"fbank": {0: "n"}, "emb": {0: "n"}}, opset_version=17,
                      dynamo=False)
    quantize_dynamic(str(fp32), str(out / "lung_sound_ast.onnx"), weight_type=QuantType.QInt8, per_channel=True)
    log.info("exported %s (%.0f MB)", out / "lung_sound_ast.onnx", (out / "lung_sound_ast.onnx").stat().st_size / 1e6)


# ---------- stage 3: int8 embeddings (onnxruntime only) ----------
def embed(out, batch=16):
    import onnxruntime as ort
    out = Path(out)
    d = np.load(out / "cycles.npz", allow_pickle=True)
    sess = ort.InferenceSession(str(out / "lung_sound_ast.onnx"), providers=["CPUExecutionProvider"])
    res, t0, n = {}, time.time(), 0
    for split in ("train", "test"):
        X = d[f"X{split}"]
        res[f"E{split}"] = np.concatenate([sess.run(None, {"fbank": to_fbank(X[i:i + batch])})[0]
                                           for i in range(0, len(X), batch)]).astype(np.float32)
        n += len(X)
    log.info("embeddings %s, %.0f ms per cycle", {k: v.shape for k, v in res.items()}, (time.time() - t0) / n * 1000)
    np.savez(out / "emb.npz", **res)


# ---------- stage 4: head (no torch) ----------
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
    import joblib
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
                                            "model": f"slim AST int8 ({N_LAYERS} layers, {CYCLE_S:g} s) + {best}",
                                            "val_scores": scores, "n_test_cycles": len(yte), "n_test_patients": len(set(pte))}
    (out / "lung_sound_ast_metrics.json").write_text(json.dumps(r, indent=2, default=float))
    joblib.dump({"model": model, "labels": list(model.classes_), "kind": "lung", "threshold": None, "positive": None,
                 "trained_on": "icbhi"}, out / "lung_sound_ast.head")
    (out / "lung_sound_ast.onnx.json").write_text(json.dumps({"role": "embedder", "for": "lung_sound",
                                                               "head": "lung_sound_ast.head"}))
    log.info("TEST ICBHI %s", r["icbhi"])
    return r


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    stage, args = sys.argv[1], sys.argv[2:]
    {"prep": prep, "export": export, "embed": embed, "head": head}[stage](*args)
