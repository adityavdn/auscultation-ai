import numpy as np

from ml.cnn_frontend import CYCLE_S, N_FRAMES, fit_length, mel32k


def test_fit_length_tiles_short_and_crops_long():
    x = np.arange(10, dtype=np.float32)
    assert list(fit_length(x, 25)[:12]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    assert len(fit_length(np.ones(100), 25)) == 25


def test_mel32k_shape_and_finite():
    x = (0.3 * np.sin(2 * np.pi * 300 * np.arange(int(2.5 * 4000)) / 4000)).astype(np.float32)
    m = mel32k(x)
    assert m.shape == (128, N_FRAMES) and np.isfinite(m).all() and m.dtype == np.float32
    assert N_FRAMES == int(CYCLE_S * 32000 / 320) + 1
