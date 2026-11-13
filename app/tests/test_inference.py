from app.inference import Analyzer
from app.tests.conftest import tone, wav_bytes


def test_heart_analysis_shape(models_dir):
    r = Analyzer(models_dir).analyze(wav_bytes(tone(220, 12)), "heart")
    assert set(r["heads"]) == {"heart_abnormal", "heart_murmur", "heart_valve"}
    h = r["heads"]["heart_abnormal"]
    assert h["label"] == "abnormal" and 0 <= h["confidence"] <= 1
    assert len(r["window_scores"]) >= 2 and r["spectrogram_png_b64"]


def test_lung_analysis_labels(models_dir):
    r = Analyzer(models_dir).analyze(wav_bytes(tone(600, 10)), "lung")
    assert r["heads"]["lung_diagnosis"]["label"] == "copd"
    assert set(r["heads"]["lung_sound"]["probs"]) == {"normal", "crackle"}
