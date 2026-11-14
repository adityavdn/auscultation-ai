"""Runs the trained heads on one recording and returns per-head results plus a spectrogram."""
import base64
import io
from pathlib import Path

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml.audio import SR, WIN, load_audio, windows  # noqa: E402
from ml.audio import AudioError  # noqa: E402
from ml.cnn_frontend import mel32k  # noqa: E402
from ml.features import logmel, stat_features  # noqa: E402
from ml.ood import is_ood  # noqa: E402

LOW_CONF = 0.6
CYCLE_WIN = (3.0, 1.5)  # lung_sound was trained on single breath cycles, so score short windows
MAX_EMBED_WINDOWS = 12  # AST costs ~0.7 s a window (int8, Apple M3 CPU), so long recordings are spread over 12
AST_SR = 16000
PRIMARY = {"heart": ("heart_abnormal", "abnormal"), "lung": ("lung_sound", "normal")}


def _cycle_windows(x, max_n=None):
    """Breath-cycle windows and their hop in seconds; with max_n the hop widens so at most max_n are cut."""
    win_s, hop_s = CYCLE_WIN
    dur = len(x) / SR
    if max_n and dur > win_s:
        hop_s = max(hop_s, (dur - win_s) / (max_n - 1))
    win, hop = int(win_s * SR), int(hop_s * SR)
    return [x[s:s + win] for s in range(0, max(len(x) - win, 0) + 1, hop)][:max_n], hop_s


def _spectrogram_png(x):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=100)
    ax.imshow(logmel(x), origin="lower", aspect="auto", cmap="magma",
              extent=[0, len(x) / SR, 0, 64])
    ax.set_xlabel("seconds")
    ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


class Analyzer:
    def __init__(self, models_dir):
        files = [p for p in sorted(Path(models_dir).glob("*.joblib")) if not p.name.startswith(".")]
        self.ood = {p.stem.removeprefix("ood_"): joblib.load(p) for p in files if p.stem.startswith("ood_")}
        self.heads = {p.stem: joblib.load(p) for p in files if not p.stem.startswith("ood_")}
        # ONNX heads (pretrained CNNs) replace the LightGBM head of the same name.
        self.onnx, self.embedders = {}, {}
        for p in sorted(Path(models_dir).glob("*.onnx")):
            if p.name.startswith("."):
                continue
            import onnxruntime as ort
            meta = json.loads(p.with_suffix(".onnx.json").read_text())
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            if meta.get("role") == "embedder":
                # Pretrained audio encoder (AST) + a small classifier head trained on its embeddings.
                self.embedders[p.stem] = sess
                self.heads[meta["for"]] = {**joblib.load(Path(models_dir) / meta["head"]), "embed": p.stem}
                continue
            self.onnx[p.stem] = (sess, meta)
            self.heads[p.stem] = {"kind": meta["kind"], "labels": meta["labels"], "trained_on": meta.get("trained_on", ""),
                                  "onnx": True}

    def _head(self, b, X):
        probs = b["model"].predict_proba(X)
        return self._summarize(b, probs), probs

    def _onnx_head(self, name, windows_):
        sess, meta = self.onnx[name]
        probs = sess.run(None, {"mel": np.stack([mel32k(w) for w in windows_])})[0]
        return self._summarize(self.heads[name], probs), probs

    def _embed_head(self, b, windows_):
        from scipy.signal import resample_poly
        from ml.ast_fbank import fbanks
        from ml.run_lung_ast import fit_cycle
        waves = [resample_poly(fit_cycle(w, SR), AST_SR // SR, 1) for w in windows_]
        probs = b["model"].predict_proba(self.embedders[b["embed"]].run(None, {"fbank": fbanks(waves)})[0])
        return self._summarize(b, probs), probs

    @staticmethod
    def _summarize(b, probs):
        mean = probs.mean(0)
        labels = list(b["labels"])
        if b.get("positive"):
            pos = labels.index(b["positive"])
            neg = 1 - pos
            label = labels[pos] if mean[pos] >= b["threshold"] else labels[neg]
            conf = mean[pos] if label == b["positive"] else 1 - mean[pos]
        else:
            label, conf = labels[int(mean.argmax())], mean.max()
        return {"label": label, "confidence": float(conf), "probs": dict(zip(labels, map(float, mean))),
                "trained_on": b.get("trained_on", ""), "low_confidence": bool(conf < LOW_CONF)}

    def analyze(self, audio, kind):
        x = load_audio(audio, kind)
        cycle, cycle_hop = _cycle_windows(x)
        segs = {"default": windows(x, kind), "cycle": cycle}
        hops = {"default": WIN[kind], "cycle": (CYCLE_WIN[0], cycle_hop)}
        feats = {k: np.array([stat_features(w, kind) for w in v]) for k, v in segs.items() if v}
        if kind in self.ood and is_ood(self.ood[kind], feats["default"]):
            raise AudioError(f"This doesn't sound like a {kind} recording from a stethoscope. "
                             "Place the stethoscope on the chest and record again.")
        heads, per_window = {}, {}
        for name, b in self.heads.items():
            if b["kind"] != kind:
                continue
            key = "cycle" if name == "lung_sound" and "cycle" in feats else "default"
            if b.get("embed"):
                key = "embed"
                segs[key], embed_hop = _cycle_windows(x, MAX_EMBED_WINDOWS)
                hops[key] = (CYCLE_WIN[0], embed_hop)
                heads[name], per_window[name] = self._embed_head(b, segs[key])
            elif b.get("onnx"):
                heads[name], per_window[name] = self._onnx_head(name, segs[key])
            else:
                heads[name], per_window[name] = self._head(b, feats[key])
            per_window[name] = (key, per_window[name])
        primary, ref = PRIMARY[kind]
        scores = []
        if primary in per_window:
            key, probs = per_window[primary]
            labels = list(self.heads[primary]["labels"])
            col = probs[:, labels.index(ref)]
            abn = col if ref != "normal" else 1 - col
            win, hop = hops[key]
            scores = [{"start": i * hop, "end": min(i * hop + win, len(x) / SR), "score": float(s)}
                      for i, s in enumerate(abn)]
        return {"kind": kind, "duration_s": len(x) / SR, "heads": heads, "window_scores": scores,
                "spectrogram_png_b64": _spectrogram_png(x)}
