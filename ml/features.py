"""Hand-crafted features (for LightGBM) and log-mel spectrograms (for the CNN)."""
import librosa
import numpy as np
from scipy.signal import hilbert
from scipy.stats import skew

from ml.audio import SR

N_FFT, HOP, N_MELS = 256, 64, 64
BAND_EDGES = {"heart": [25, 50, 100, 150, 200, 300, 400], "lung": [50, 100, 200, 400, 600, 800, 1200, 1900]}


def logmel(x):
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=2000)
    return librosa.power_to_db(m + 1e-10).astype(np.float32)


def _stats(m):
    return np.concatenate([m.mean(1), m.std(1), np.nan_to_num(skew(m, axis=1))])


def _band_ratios(x, kind):
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    edges = BAND_EDGES[kind]
    e = np.array([spec[(freqs >= a) & (freqs < b)].sum() for a, b in zip(edges, edges[1:])])
    return e / (e.sum() + 1e-12)


def _periodicity(x):
    """Heart-cycle regularity: autocorrelation peak of the envelope at 0.4-1.5 s lag."""
    env = np.abs(hilbert(x))[::40]  # 100 Hz envelope
    env = env - env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    seg = ac[40:150]
    return np.array([seg.max() if len(seg) else 0.0, (np.argmax(seg) + 40) / 100 if len(seg) else 0.0])


def stat_features(x, kind):
    mf = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=40, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=2000)
    d = librosa.feature.delta(mf)
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    spec = np.vstack([
        librosa.feature.spectral_centroid(S=S, sr=SR),
        librosa.feature.spectral_bandwidth(S=S, sr=SR),
        librosa.feature.spectral_rolloff(S=S, sr=SR),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.zero_crossing_rate(x, frame_length=N_FFT, hop_length=HOP),
    ])
    parts = [_stats(mf), _stats(d), _stats(spec), _band_ratios(x, kind)]
    if kind == "heart":
        parts.append(_periodicity(x))
    return np.nan_to_num(np.concatenate(parts)).astype(np.float32)
