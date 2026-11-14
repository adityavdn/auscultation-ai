"""Tiny synthetic models with the real head names, so app tests run without Kaggle output."""
import io
import os

# The server image has no torch; keep transformers from importing it here (torch beside joblib hangs).
os.environ.setdefault("USE_TORCH", "0")

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ml.train_baseline import train_task

HEADS = {
    "heart_abnormal": ("heart", ["normal", "abnormal"], dict(positive="abnormal")),
    "heart_murmur": ("heart", ["absent", "present"], dict(positive="present")),
    "heart_valve": ("heart", ["normal", "AS"], {}),
    "lung_sound": ("lung", ["normal", "crackle"], {}),
    "lung_diagnosis": ("lung", ["healthy", "copd"], {}),
}


def tone(freq, seconds=6, sr=4000, seed=0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.default_rng(seed).standard_normal(len(t))).astype(np.float32)


def wav_bytes(x, sr=4000):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    for name, (kind, labels, kw) in HEADS.items():
        rows = []
        for i in range(24):
            f = (80 if kind == "heart" else 150) if i % 2 else (220 if kind == "heart" else 600)
            p = d / f"{name}_{i}.wav"
            sf.write(p, tone(f, seed=i), 4000)
            rows.append({"path": str(p), "patient_id": f"p{i}", "label": labels[i % 2 == 0],
                         "dataset": "synth", "split": "test" if i < 6 else "train"})
        train_task(pd.DataFrame(rows), kind, name, d, **kw)
    return d
