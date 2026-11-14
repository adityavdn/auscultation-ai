import numpy as np

from ml.ood import fit_ood, is_ood


def test_far_input_flagged_and_training_like_input_not():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (2000, 20))
    b = fit_ood(X)
    assert not is_ood(b, rng.normal(0, 1, (5, 20)))
    assert is_ood(b, rng.normal(0, 1, (5, 20)) + 15)


def test_train_ood_writes_gate_from_train_split_only(tmp_path):
    import joblib
    import pandas as pd
    import soundfile as sf

    from ml.ood import train_ood
    rows = []
    for i in range(10):
        p = tmp_path / f"r{i}.wav"
        t = np.arange(4000 * 8) / 4000
        sf.write(p, (0.5 * np.sin(2 * np.pi * 120 * t) + 0.05 * np.random.default_rng(i).standard_normal(len(t))).astype(np.float32), 4000)
        rows.append({"path": str(p), "patient_id": f"p{i}", "label": "x", "dataset": "s", "split": "test" if i < 2 else "train"})
    n = train_ood([pd.DataFrame(rows)], "heart", tmp_path)
    b = joblib.load(tmp_path / "ood_heart.joblib")
    assert n == 8 * 3 and "threshold" in b  # 8 train recordings x 3 windows (0, 2.5, 5 s); test rows unused
