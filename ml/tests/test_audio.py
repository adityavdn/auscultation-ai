import io

import numpy as np
import pytest
import soundfile as sf

from ml.audio import SR, AudioError, load_audio, windows
from ml.features import logmel, stat_features


def _wav_bytes(x, sr):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


def test_resample_and_mono():
    t = np.arange(int(3 * 44100)) / 44100
    s = 0.5 * np.sin(2 * np.pi * 100 * t)
    x = load_audio(_wav_bytes(np.stack([s, s], 1), 44100), "heart")
    assert x.ndim == 1 and len(x) == 3 * SR


def test_short_raises():
    with pytest.raises(AudioError):
        load_audio(_wav_bytes(np.random.randn(int(1.5 * 4000)) * 0.1, 4000), "lung")


def test_silent_raises():
    with pytest.raises(AudioError):
        load_audio(_wav_bytes(np.zeros(4000 * 3), 4000), "heart")


def test_windows_count():
    assert len(windows(np.ones(12 * SR, dtype=np.float32), "heart")) == 4


def test_stat_features_finite_and_fixed_length():
    rng = np.random.default_rng(0)
    a = stat_features(rng.standard_normal(5 * SR).astype(np.float32), "heart")
    b = stat_features(rng.standard_normal(5 * SR).astype(np.float32), "heart")
    assert np.isfinite(a).all() and a.shape == b.shape


def test_logmel_shape():
    assert logmel(np.zeros(SR * 5, dtype=np.float32) + 0.01).shape[0] == 64


def test_short_recording_still_gives_one_window():
    w = windows(np.ones(int(2.1 * SR), dtype=np.float32), "heart")
    assert len(w) == 1 and len(w[0]) == 5 * SR


def test_m4a_with_moov_at_end_decodes(tmp_path):
    import subprocess
    src = tmp_path / "in.wav"
    src.write_bytes(_wav_bytes((0.5 * np.sin(2 * np.pi * 100 * np.arange(90 * 44100) / 44100)).astype(np.float32), 44100))
    out = tmp_path / "out.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "aac", str(out)], check=True)
    assert len(load_audio(out.read_bytes(), "heart")) >= 80 * SR


def test_undecodable_says_could_not_decode():
    with pytest.raises(AudioError, match="decode"):
        load_audio(b"\x00" * 5000, "heart")


def test_nan_audio_rejected():
    x = (0.3 * np.sin(np.arange(4000 * 4) / 3)).astype(np.float32)
    x[100] = np.nan
    buf = io.BytesIO()
    sf.write(buf, x, 4000, format="WAV", subtype="FLOAT")
    with pytest.raises(AudioError):
        load_audio(buf.getvalue(), "heart")


def test_clipped_audio_rejected():
    x = np.clip(3 * np.sin(2 * np.pi * 100 * np.arange(4000 * 4) / 4000), -1, 1).astype(np.float32)
    with pytest.raises(AudioError, match="clipped|loud"):
        load_audio(_wav_bytes(x, 4000), "heart")


def test_long_audio_truncated_to_cap():
    from ml.audio import MAX_SECONDS
    x = (0.3 * np.random.default_rng(0).standard_normal(4000 * (MAX_SECONDS + 30))).astype(np.float32)
    assert len(load_audio(_wav_bytes(x, 4000), "lung")) == MAX_SECONDS * SR
