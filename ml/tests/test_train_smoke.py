import numpy as np
import pandas as pd
import soundfile as sf
import joblib

from ml.train_baseline import train_task


def test_train_task_separates_two_tones(tmp_path):
    rows = []
    for i in range(24):
        f, lab = (80, "normal") if i % 2 else (220, "abnormal")
        t = np.arange(4000 * 6) / 4000
        x = 0.5 * np.sin(2 * np.pi * f * t) + 0.05 * np.random.default_rng(i).standard_normal(len(t))
        p = tmp_path / f"r{i}.wav"
        sf.write(p, x.astype(np.float32), 4000)
        rows.append({"path": str(p), "patient_id": f"p{i}", "label": lab, "dataset": "synth",
                     "split": "test" if i < 6 else "train"})
    r = train_task(pd.DataFrame(rows), "heart", "smoke", tmp_path / "models", positive="abnormal")
    assert r["macro_f1"] > 0.9
    bundle = joblib.load(tmp_path / "models/smoke.joblib")
    assert set(bundle) >= {"model", "labels", "kind", "threshold", "positive"}
