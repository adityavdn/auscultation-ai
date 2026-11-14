"""Log-mel front end for the pretrained lung CNN, shared by training and serving (numpy + librosa only).

The 4 kHz signal is upsampled to 32 kHz so the 128 mel bands keep the AudioSet layout the pretrained
EfficientAT network expects; only bands below 2 kHz carry energy, consistently for every device.
"""
import librosa
import numpy as np
from scipy.signal import resample_poly

from ml.audio import SR

CYCLE_S = 4.0          # every breath cycle is tiled or cropped to this length
TARGET_SR = 32000
HOP = 320
N_FRAMES = int(CYCLE_S * TARGET_SR / HOP) + 1


def fit_length(x, n):
    """Repeat a short signal (or crop a long one) to exactly n samples."""
    return np.resize(np.asarray(x, dtype=np.float32), n)


def mel32k(x4k):
    x = resample_poly(fit_length(x4k, int(CYCLE_S * SR)), TARGET_SR // SR, 1).astype(np.float32)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])  # pre-emphasis, as in EfficientAT
    m = librosa.feature.melspectrogram(y=x, sr=TARGET_SR, n_fft=1024, win_length=800, hop_length=HOP,
                                       n_mels=128, fmin=0, fmax=15000, htk=True, power=2.0)
    return ((np.log(m + 1e-5) + 4.5) / 5.0).astype(np.float32)
