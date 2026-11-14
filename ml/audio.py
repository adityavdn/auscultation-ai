"""Audio loading: any format -> 4 kHz mono, band-passed, peak-normalised."""
import io
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 4000
BANDS = {"heart": (25, 400), "lung": (50, 1900)}
WIN = {"heart": (5.0, 2.5), "lung": (8.0, 4.0)}  # (window s, hop s)
MIN_SECONDS = 2.0
MIN_RMS = 1e-4
MAX_SECONDS = 120  # longer uploads are cut to the first two minutes
CLIP_FRACTION = 0.10  # reject when >10% of samples sit in flat runs at the peak; real data 99th pct is ~6%


class AudioError(ValueError):
    """Audio unusable: too short, silent or undecodable."""


def _decode(src):
    """Return (samples[n] or [n, ch], sr). Falls back to ffmpeg for webm/m4a/mp3."""
    try:
        return sf.read(io.BytesIO(src) if isinstance(src, bytes) else src, dtype="float32")
    except Exception:
        pass
    # ffmpeg needs a seekable file: m4a often stores its index (moov atom) at the end.
    with tempfile.NamedTemporaryFile() as tmp:
        if isinstance(src, bytes):
            tmp.write(src)
            tmp.flush()
        cmd = ["ffmpeg", "-v", "error", "-i", tmp.name if isinstance(src, bytes) else str(src),
               "-t", str(MAX_SECONDS), "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
        try:
            out = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise AudioError("Could not decode audio file.") from e
    if not out:
        raise AudioError("Could not decode audio file.")
    return np.frombuffer(out, dtype=np.float32), SR


def _clipped(x):
    peak = np.max(np.abs(x))
    if peak == 0:
        return False
    at_peak = np.abs(x) >= 0.999 * peak
    flat = at_peak[1:] & at_peak[:-1] & (np.abs(np.diff(x)) < 1e-6)
    return flat.mean() > CLIP_FRACTION


def load_audio(src, kind):
    x, sr = _decode(src)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if not np.isfinite(x).all():
        raise AudioError("Recording contains invalid samples; record again.")
    if _clipped(x):
        raise AudioError("Recording is clipped (too loud); lower the stethoscope volume and record again.")
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    x = x[:MAX_SECONDS * SR]
    if len(x) < MIN_SECONDS * SR:
        raise AudioError(f"Recording too short ({len(x) / SR:.1f}s); need at least {MIN_SECONDS:.0f}s.")
    lo, hi = BANDS[kind]
    x = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=SR, output="sos"), x)
    if np.sqrt(np.mean(x ** 2)) < MIN_RMS:
        raise AudioError("Recording is silent; check the stethoscope contact.")
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def windows(x, kind, sr=SR):
    """Fixed windows; a final partial window is zero-padded if at least half full.
    A recording shorter than half a window still yields one zero-padded window."""
    win, hop = (int(v * sr) for v in WIN[kind])
    out = []
    for start in range(0, max(len(x), 1), hop):
        seg = x[start:start + win]
        if len(seg) < win / 2 and out:
            break
        out.append(np.pad(seg, (0, win - len(seg))))
        if start + win >= len(x):
            break
    return out
