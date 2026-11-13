"""Bundle ml/ into one self-contained Kaggle script: kaggle/build/run.py + kernel-metadata.json."""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle" / "build"


def main(username, slug="auscultation-baseline", gpu=False):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted((ROOT / "ml").glob("*.py")):
            tar.add(f, arcname=f"ml/{f.name}")
    payload = base64.b64encode(buf.getvalue()).decode()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.py").write_text(f'''import base64, io, subprocess, sys, tarfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "librosa", "lightgbm", "soundfile"], check=False)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{payload}")), mode="r:gz").extractall("/kaggle/working/src")
sys.path.insert(0, "/kaggle/working/src")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
from ml.run_all import run
run("/kaggle/tmp/data", "/kaggle/working/out")
''')
    (OUT / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": ["vbookshelf/respiratory-sound-database"], "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
