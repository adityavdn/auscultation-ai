"""AST input features in numpy only (the transformers extractor pads with torch, which the server image lacks).

Matches ASTFeatureExtractor's no-torchaudio path: Kaldi-style log-mel fbank, padded or cut to 1024 frames,
then normalised with the AudioSet mean and std.
"""
import numpy as np
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

SR = 16000
N_MELS = 128
MAX_FRAMES = 1024
MEAN, STD = -4.2677393, 4.5689974
_MEL = mel_filter_bank(num_frequency_bins=257, num_mel_filters=N_MELS, min_frequency=20, max_frequency=SR // 2,
                       sampling_rate=SR, norm=None, mel_scale="kaldi", triangularize_in_mel_space=True)
_WINDOW = window_function(400, "hann", periodic=False)


def fbank(wave, max_frames=MAX_FRAMES):
    """One 16 kHz waveform -> (max_frames, 128) float32 normalised log-mel frames."""
    f = spectrogram(np.squeeze(np.asarray(wave, dtype=np.float32)), _WINDOW, frame_length=400, hop_length=160,
                    fft_length=512, power=2.0, center=False, preemphasis=0.97, mel_filters=_MEL, log_mel="log",
                    mel_floor=1.192092955078125e-07, remove_dc_offset=True).T
    out = np.zeros((max_frames, N_MELS), dtype=np.float32)
    n = min(len(f), max_frames)
    out[:n] = f[:n]
    return (out - MEAN) / (STD * 2)


def fbanks(waves, max_frames=MAX_FRAMES):
    return np.stack([fbank(w, max_frames) for w in waves]).astype(np.float32)
