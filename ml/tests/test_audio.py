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
