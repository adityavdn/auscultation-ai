import os

import numpy as np
import pytest

# torch in the same process as the joblib/LightGBM tests hangs on macOS; run these on their own:
#   RUN_TORCH_TESTS=1 pytest ml/tests/test_lung_cnn.py
pytestmark = pytest.mark.skipif(not os.environ.get("RUN_TORCH_TESTS"), reason="set RUN_TORCH_TESTS=1")
torch = pytest.importorskip("torch")

from ml.lung_cnn import build_model, export_onnx, fit  # noqa: E402


def _toy(n=48, t=101, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.1, (n, 128, t)).astype(np.float32)
    y = np.array(["a", "b"] * (n // 2))
    X[y == "b", :64, :] += 3.0  # class b: strong energy across the lower half of the bands
    return X, y


def test_forward_shape():
    m = build_model(4, pretrained=False)
    assert m(torch.zeros(3, 128, 101)).shape == (3, 4)


def test_fit_learns_toy_task_and_onnx_matches(tmp_path):
    X, y = _toy()
    Xv, yv = _toy(16, seed=1)
    model, info = fit(X, y, Xv, yv, labels=["a", "b"], normal_label="a", epochs=3, batch=16, lr=1e-3,
                      pretrained=True, device="cpu")
    assert info["best_val_score"] > 0.9
    p = export_onnx(model, tmp_path / "m.onnx", n_frames=101)
    import onnxruntime as ort
    out = ort.InferenceSession(str(p)).run(None, {"mel": Xv[:5]})[0]
    with torch.no_grad():
        ref = torch.softmax(model.eval()(torch.from_numpy(Xv[:5])), 1).numpy()
    assert np.allclose(out, ref, atol=1e-4)
