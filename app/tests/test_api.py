import json
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import DISCLAIMER, create_app
from app.tests.conftest import tone, wav_bytes


@pytest.fixture()
def client(models_dir, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "s1.wav").write_bytes(wav_bytes(tone(220, 8)))
    (samples / "manifest.json").write_text(json.dumps([{"id": "s1", "kind": "heart", "task": "heart_abnormal",
                                                         "true_label": "abnormal", "file": "s1.wav",
                                                         "dataset": "synth"}]))
    return TestClient(create_app(models_dir, samples, tmp_path / "db.sqlite"))


def post(client, kind, data, name="a.wav"):
    return client.post(f"/api/analyze/{kind}", files={"file": (name, data)})


def test_analyze_wav_ok(client):
    r = post(client, "heart", wav_bytes(tone(220, 8)))
    assert r.status_code == 200 and r.json()["heads"]["heart_abnormal"]["label"] == "abnormal"


def test_disclaimer_present(client):
    assert post(client, "heart", wav_bytes(tone(220, 8))).json()["disclaimer"] == DISCLAIMER


def test_analyze_webm_ok(client, tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(wav_bytes(tone(220, 8)))
    out = tmp_path / "out.webm"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "libopus", str(out)], check=True)
    r = post(client, "heart", out.read_bytes(), "rec.webm")
    assert r.status_code == 200, r.text


def test_short_audio_422(client):
    r = post(client, "lung", wav_bytes(tone(600, 1)))
    assert r.status_code == 422 and "short" in r.json()["detail"]


def test_garbage_audio_422(client):
    assert post(client, "lung", b"not audio at all").status_code == 422


def test_too_big_413(client):
    assert post(client, "heart", b"\0" * (20 * 1024 * 1024 + 1)).status_code == 413


def test_unknown_kind_404(client):
    assert post(client, "kidney", wav_bytes(tone(220, 8))).status_code == 404


def test_combined_requires_one(client):
    assert client.post("/api/analyze/combined").status_code == 422


def test_combined_fuses(client):
    r = client.post("/api/analyze/combined", files={"heart": ("h.wav", wav_bytes(tone(220, 8))),
                                                    "lung": ("l.wav", wav_bytes(tone(600, 8)))})
    body = r.json()
    assert r.status_code == 200 and body["fusion"]["status"] in {"normal", "abnormal", "inconclusive"}
    assert body["heart"] and body["lung"]


def test_samples_list_and_audio(client):
    s = client.get("/api/samples").json()
    assert s[0]["id"] == "s1" and "file" not in s[0]
    assert client.get("/api/samples/s1/audio").status_code == 200
    assert client.get("/api/samples/../../etc/passwd/audio").status_code == 404


def test_cases_roundtrip(client):
    c = client.post("/api/cases", json={"patient_name": "Test", "summary": {"symptoms": "cough"},
                                        "analysis": {"status": "abnormal"}, "urgency": "soon"}).json()
    assert client.get(f"/api/cases/{c['id']}").json()["summary"]["symptoms"] == "cough"
    assert any(x["id"] == c["id"] for x in client.get("/api/cases").json())


def test_case_bad_urgency_422(client):
    assert client.post("/api/cases", json={"summary": {}, "urgency": "whenever"}).status_code == 422
