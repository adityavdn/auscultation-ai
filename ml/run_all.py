"""Download all datasets, train every task, write models/, reports/metrics.json and samples/.

Usage: python -m ml.run_all <data_dir> <out_dir>
"""
import json
import os
import logging
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np

from ml import datasets as D
from ml.metrics import macc
from ml.train_baseline import _rec_proba, featurize, train_task

log = logging.getLogger("run_all")
URLS = {
    "cinc": "https://physionet.org/files/challenge-2016/1.0.0/training.zip",
    "circor": "https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/",
    "kauh": "https://data.mendeley.com/public-files/datasets/jwyy9np4gv/files/99d7bd63-eb5d-4000-9c2a-a8dd31168cbc/file_downloaded",
}
OPERA = "https://raw.githubusercontent.com/evelyn0414/OPERA/main/datasets/icbhi/"
YASEEN = ("https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-"
          "/raw/master/{}_New_3%EC%A3%BC%EA%B8%B0.rar")
# Official ICBHI site was down (13 Nov 2026): audio from an HF mirror that keeps original filenames,
# cycle annotations from a public GitHub copy of the official .txt files.
ICBHI_HF = "https://huggingface.co/datasets/CoughMamba/icbhi/resolve/main/data/train-0000{}-of-00005.parquet"
ICBHI_TXT_REPO = "https://github.com/Wangfengyi0116/ICBHIdatabase.git"
SAMPLES_PER_CLASS = 3


def _get(url, dest):
    dest = Path(dest)
    if not dest.exists():
        log.info("download %s", url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _unzip(zf, to):
    to = Path(to)
    if not to.exists():
        zipfile.ZipFile(zf).extractall(to)
    return to


def _unrar(rar, to):
    to.mkdir(parents=True, exist_ok=True)
    for cmd in (["unrar", "x", "-o+", str(rar), str(to) + "/"], ["7z", "x", "-y", f"-o{to}", str(rar)],
                ["bsdtar", "-xf", str(rar), "-C", str(to)]):
        if shutil.which(cmd[0]) and subprocess.run(cmd, capture_output=True, env=os.environ | {"LC_ALL": "en_US.UTF-8"}).returncode == 0:
            return
    subprocess.run("apt-get install -y -qq unrar-free p7zip-full >/dev/null 2>&1", shell=True)
    subprocess.run(["7z", "x", "-y", f"-o{to}", str(rar)], check=True, capture_output=True)


def _find(root, pattern):
    hits = sorted(Path(root).rglob(pattern))
    return hits[0] if hits else None


def fetch_icbhi(data):
    import pyarrow.parquet as pq
    out = Path(data) / "icbhi"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.wav"))) < 920:
        for i in range(5):
            shard = _get(ICBHI_HF.format(i), Path(data) / f"icbhi_{i}.parquet")
            for row in pq.read_table(shard, columns=["audio"]).to_pylist():
                (out / Path(row["audio"]["path"]).name).write_bytes(row["audio"]["bytes"])
            shard.unlink()
    if len(list(out.glob("*.txt"))) < 920:
        repo = Path(data) / "icbhi_txt"
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", ICBHI_TXT_REPO, str(repo)], check=True, capture_output=True)
        for t in (repo / "data").glob("*_*_*_*_*.txt"):
            shutil.copy(t, out / t.name)
    return out


def fetch_all(data):
    data = Path(data)
    paths = {}
    cinc = _unzip(_get(URLS["cinc"], data / "cinc.zip"), data / "cinc")
    paths["cinc"] = _find(cinc, "REFERENCE.csv").parent.parent
    circor = _unzip(_get(URLS["circor"], data / "circor.zip"), data / "circor")
    paths["circor"] = _find(circor, "training_data.csv").parent
    kauh = _unzip(_get(URLS["kauh"], data / "kauh.zip"), data / "kauh")
    paths["kauh"] = kauh
    for c in ("N", "AS", "MS", "MR", "MVP"):
        rar = _get(YASEEN.format(c), data / f"yaseen_{c}.rar")
        dest = data / "yaseen" / f"{c}_New"
        if not dest.exists():
            _unrar(rar, dest)
    paths["yaseen"] = data / "yaseen"
    paths["icbhi_split"] = _get(OPERA + "ICBHI_challenge_train_test.txt", data / "icbhi_split.txt")
    paths["icbhi_diag"] = _get(OPERA + "ICBHI_Challenge_diagnosis.txt", data / "icbhi_diag.txt")
    wav = _find("/kaggle/input", "101_1b1_Al_sc_Meditron.wav") if Path("/kaggle/input").exists() else None
    paths["icbhi"] = wav.parent if wav else fetch_icbhi(data)
    return paths


def _save_samples(df, task, kind, out):
    rows = []
    test = df[df.split == "test"].drop_duplicates("path")
    for label, g in test.groupby("label"):
        for _, r in g.head(SAMPLES_PER_CLASS).iterrows():
            name = f"{task}__{label}__{Path(r['path']).stem}.wav".replace(" ", "_")
            shutil.copy(r["path"], out / name)
            rows.append({"id": Path(name).stem, "kind": kind, "task": task, "true_label": label,
                         "file": name, "dataset": r["dataset"]})
    return rows


def cross_dataset(models, circor_df):
    """Adult (CinC 2016) abnormal model tested on paediatric CirCor: murmur present ~ abnormal."""
    b = joblib.load(models / "heart_abnormal.joblib")
    te = circor_df[circor_df.split == "test"].reset_index(drop=True)
    X, _, rec, _ = featurize(te, "heart")
    p, idx = _rec_proba(b["model"], X, rec)
    pred = p[:, b["labels"].index("abnormal")] >= b["threshold"]
    return {"macc": macc(list(te["label"].iloc[idx] == "present"), list(pred), positive=True),
            "note": "train CinC2016 (adult) -> test CirCor (paediatric), murmur present treated as abnormal"}


def run(data, out):
    out = Path(out)
    models, samples = out / "models", out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    p = fetch_all(data)
    tasks = {
        "heart_abnormal": (lambda: D.load_cinc2016(p["cinc"]), "heart", dict(positive="abnormal")),
        "heart_murmur": (lambda: D.load_circor(p["circor"]), "heart", dict(positive="present")),
        "heart_valve": (lambda: D.load_yaseen(p["yaseen"]), "heart", {}),
        "lung_sound": (lambda: D.load_icbhi_cycles(p["icbhi"], p["icbhi_split"]), "lung",
                       dict(normal_label="normal")),
        "lung_diagnosis": (lambda: D.load_lung_diagnosis(p["icbhi"], p["icbhi_diag"], p["kauh"]), "lung", {}),
    }
    metrics, sample_rows, frames = {}, [], {}
    for name, (load, kind, kw) in tasks.items():
        try:
            df = frames[name] = load()
            log.info("%s: %d rows, %s", name, len(df), df.groupby(["split", "label"]).size().to_dict())
            metrics[name] = train_task(df, kind, name, models, **kw)
            if name != "lung_sound":
                sample_rows += _save_samples(df, name, kind, samples)
        except Exception as e:  # keep going: one broken dataset must not lose the others
            log.error("%s failed: %s", name, traceback.format_exc())
            metrics[name] = {"error": repr(e)}
    try:
        metrics["cross_cinc_to_circor"] = cross_dataset(models, frames["heart_murmur"])
    except Exception as e:
        metrics["cross_cinc_to_circor"] = {"error": repr(e)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (samples / "manifest.json").write_text(json.dumps(sample_rows, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    np.seterr(all="ignore")
    run(sys.argv[1], sys.argv[2])
