"""Audio loading: any format -> 4 kHz mono, band-passed, peak-normalised."""
import io
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 4000
BANDS = {"heart": (25, 400), "lung": (50, 1900)}
WIN = {"heart": (5.0, 2.5), "lung": (8.0, 4.0)}  # (window s, hop s)
MIN_SECONDS = 2.0
MIN_RMS = 1e-4


class AudioError(ValueError):
    """Audio unusable: too short, silent or undecodable."""


def _decode(src):
    """Return (samples[n] or [n, ch], sr). Falls back to ffmpeg for webm/m4a/mp3."""
    try:
        return sf.read(io.BytesIO(src) if isinstance(src, bytes) else src, dtype="float32")
    except Exception:
        pass
    cmd = ["ffmpeg", "-v", "error", "-i", "pipe:0" if isinstance(src, bytes) else str(src),
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    try:
        out = subprocess.run(cmd, input=src if isinstance(src, bytes) else None,
                             capture_output=True, check=True, timeout=60).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise AudioError("Could not decode audio file.") from e
    return np.frombuffer(out, dtype=np.float32), SR


def load_audio(src, kind):
    x, sr = _decode(src)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    if len(x) < MIN_SECONDS * SR:
        raise AudioError(f"Recording too short ({len(x) / SR:.1f}s); need at least {MIN_SECONDS:.0f}s.")
    lo, hi = BANDS[kind]
    x = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=SR, output="sos"), x)
    if np.sqrt(np.mean(x ** 2)) < MIN_RMS:
        raise AudioError("Recording is silent; check the stethoscope contact.")
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def windows(x, kind, sr=SR):
    """Fixed windows; a final partial window is zero-padded if at least half full."""
    win, hop = (int(v * sr) for v in WIN[kind])
    out = []
    for start in range(0, max(len(x), 1), hop):
        seg = x[start:start + win]
        if len(seg) < win / 2:
            break
        out.append(np.pad(seg, (0, win - len(seg))))
        if start + win >= len(x):
            break
    return out
