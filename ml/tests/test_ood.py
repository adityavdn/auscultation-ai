import numpy as np

from ml.ood import fit_ood, is_ood


def test_far_input_flagged_and_training_like_input_not():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (2000, 20))
    b = fit_ood(X)
    assert not is_ood(b, rng.normal(0, 1, (5, 20)))
    assert is_ood(b, rng.normal(0, 1, (5, 20)) + 15)
