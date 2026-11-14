from pathlib import Path

from ml import datasets as D


def _touch(p: Path, text=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _no_leak(df):
    tr, te = df[df.split == "train"], df[df.split == "test"]
    assert len(tr) and len(te)
    assert set(tr.patient_id).isdisjoint(te.patient_id)
    assert {"path", "patient_id", "label", "dataset", "split"} <= set(df.columns)


def test_cinc2016(tmp_path):
    lines = "\n".join(f"a{i:04d},{-1 if i % 2 else 1}" for i in range(20))
    _touch(tmp_path / "training-a/REFERENCE.csv", lines)
    df = D.load_cinc2016(tmp_path)
    _no_leak(df)
    assert set(df.label) == {"normal", "abnormal"}


def test_circor_drops_unknown_and_groups_patient(tmp_path):
    rows = ["Patient ID,Locations,Age,Sex,Murmur,Outcome"]
    for pid in range(100, 120):
        m = ["Present", "Absent", "Unknown"][pid % 3]
        rows.append(f"{pid},AV+MV,Child,Male,{m},Normal")
        for loc in ("AV", "MV"):
            _touch(tmp_path / f"training_data/{pid}_{loc}.wav")
    _touch(tmp_path / "training_data.csv", "\n".join(rows))
    df = D.load_circor(tmp_path)
    _no_leak(df)
    assert set(df.label) == {"present", "absent"}
    assert len(df) == 2 * sum(1 for p in range(100, 120) if p % 3 != 2)


def test_yaseen_labels(tmp_path):
    for folder, n in [("N_New_3주기", "N"), ("AS_New_3주기", "AS"), ("MVP_New_3주기", "MVP")]:
        for i in range(10):
            _touch(tmp_path / folder / f"New_{n}_{i:03d}.wav")
    df = D.load_yaseen(tmp_path)
    _no_leak(df)
    assert set(df.label) == {"normal", "AS", "MVP"}


def test_icbhi_cycle_labels_and_official_split(tmp_path):
    stems = []
    for pid in range(101, 111):
        stem = f"{pid}_1b1_Al_sc_Meditron"
        stems.append(stem)
        _touch(tmp_path / f"{stem}.wav")
        _touch(tmp_path / f"{stem}.txt", "0.1\t1.2\t1\t1\n1.2\t2.5\t0\t0\n")
    split = "\n".join(f"{s}\t{'test' if i < 4 else 'train'}" for i, s in enumerate(stems))
    _touch(tmp_path / "split.txt", split)
    df = D.load_icbhi_cycles(tmp_path, tmp_path / "split.txt")
    _no_leak(df)
    first = df[df.path.str.contains("101_")].sort_values("start")
    assert list(first.label) == ["both", "normal"]
    assert (df[df.patient_id == "icbhi_101"].split == "test").all()


def test_lung_diagnosis_merges_icbhi_and_kauh(tmp_path):
    ic, ka = tmp_path / "icbhi", tmp_path / "kauh"
    diags = {101: "URTI", 102: "Healthy", 103: "COPD", 104: "Pneumonia", 105: "Healthy",
             106: "COPD", 107: "URTI", 108: "COPD", 109: "Healthy", 110: "COPD"}
    for pid in diags:
        _touch(ic / f"{pid}_1b1_Al_sc_Meditron.wav")
    _touch(tmp_path / "diag.txt", "\n".join(f"{k}\t{v}" for k, v in diags.items()))
    for i, d in enumerate(["Asthma", "heart failure", "N", "COPD", "pneumonia"] * 2, start=1):
        _touch(ka / f"BP{i}_{d},I E W,P L L,70,M.wav")
        _touch(ka / f"EP{i}_{d},I E W,P L R,70,M.wav")
    df = D.load_lung_diagnosis(ic, tmp_path / "diag.txt", ka)
    _no_leak(df)
    assert set(df.label) <= {"healthy", "copd", "asthma", "heart_failure", "infection", "other"}
    assert {"infection", "healthy", "copd", "asthma", "heart_failure"} <= set(df.label)
    assert df[df.path.str.contains("BP1_")].patient_id.iloc[0] == df[df.path.str.contains("EP1_")].patient_id.iloc[0]
