from app.inference import Analyzer
from app.tests.conftest import tone, wav_bytes


def test_heart_analysis_shape(models_dir):
    r = Analyzer(models_dir).analyze(wav_bytes(tone(220, 12)), "heart")
    assert set(r["heads"]) == {"heart_abnormal", "heart_murmur", "heart_valve"}
    h = r["heads"]["heart_abnormal"]
    assert h["label"] == "abnormal" and 0 <= h["confidence"] <= 1
    assert len(r["window_scores"]) >= 2 and r["spectrogram_png_b64"]


def test_lung_analysis_labels(models_dir):
    r = Analyzer(models_dir).analyze(wav_bytes(tone(600, 10)), "lung")
    assert r["heads"]["lung_diagnosis"]["label"] == "copd"
    assert set(r["heads"]["lung_sound"]["probs"]) == {"normal", "crackle"}


def test_hidden_metadata_files_are_ignored(models_dir, tmp_path):
    import shutil
    d = tmp_path / "m"
    shutil.copytree(models_dir, d)
    (d / "._heart_abnormal.joblib").write_bytes(b"\x00\x05\x16\x07macOS metadata")
    assert "._heart_abnormal" not in Analyzer(d).heads


def test_ood_gate_rejects_unlike_audio(models_dir, tmp_path):
    import shutil

    import joblib
    import numpy as np
    import pytest

    from ml.audio import AudioError, load_audio, windows
    from ml.features import stat_features
    from ml.ood import fit_ood
    d = tmp_path / "m"
    shutil.copytree(models_dir, d)
    X = np.array([stat_features(w, "heart") for f in (80, 220) for s in range(8)
                  for w in windows(load_audio(wav_bytes(tone(f, 12, seed=s)), "heart"), "heart")])
    joblib.dump(fit_ood(X), d / "ood_heart.joblib")
    a = Analyzer(d)
    assert "ood_heart" not in a.heads
    a.analyze(wav_bytes(tone(220, 12, seed=99)), "heart")
    noise = np.random.default_rng(1).standard_normal(4000 * 12).astype("float32")
    with pytest.raises(AudioError, match="stethoscope"):
        a.analyze(wav_bytes(noise), "heart")


def _tiny_onnx(path, n_classes=4):
    """Torch-free stand-in for the exported CNN: mean over time, linear layer, softmax."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.random.default_rng(0).normal(0, 0.1, (128, n_classes)).astype("float32"), "w")
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["mel", "axes"], ["pooled"], keepdims=0),
         helper.make_node("MatMul", ["pooled", "w"], ["logits"]),
         helper.make_node("Softmax", ["logits"], ["probs"], axis=1)],
        "tiny", [helper.make_tensor_value_info("mel", TensorProto.FLOAT, ["batch", 128, "t"])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", n_classes])],
        [w, numpy_helper.from_array(np.array([2], dtype="int64"), "axes")])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def test_onnx_lung_sound_head_replaces_joblib(models_dir, tmp_path):
    import json
    import shutil
    d = tmp_path / "m"
    shutil.copytree(models_dir, d)
    labels = ["both", "crackle", "normal", "wheeze"]
    _tiny_onnx(d / "lung_sound.onnx")
    (d / "lung_sound.onnx.json").write_text(json.dumps({"labels": labels, "kind": "lung", "trained_on": "icbhi"}))
    r = Analyzer(d).analyze(wav_bytes(tone(600, 10)), "lung")
    h = r["heads"]["lung_sound"]
    assert set(h["probs"]) == set(labels) and h["trained_on"] == "icbhi"
    assert abs(sum(h["probs"].values()) - 1) < 1e-4 and r["window_scores"]


def _tiny_embedder(path, dim=8):
    """Torch-free stand-in for the exported AST embedder: fbank (n, frames, 128) -> mean over time -> linear."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.random.default_rng(1).normal(0, 0.1, (128, dim)).astype("float32"), "w")
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["fbank", "axes"], ["pooled"], keepdims=0),
         helper.make_node("MatMul", ["pooled", "w"], ["emb"])],
        "tiny_ast", [helper.make_tensor_value_info("fbank", TensorProto.FLOAT, ["n", "t", 128])],
        [helper.make_tensor_value_info("emb", TensorProto.FLOAT, ["n", dim])],
        [w, numpy_helper.from_array(np.array([1], dtype="int64"), "axes")])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def _ast_models(models_dir, tmp_path, labels):
    import json
    import shutil
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    d = tmp_path / "m"
    shutil.copytree(models_dir, d)
    _tiny_embedder(d / "lung_sound_ast.onnx")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 8)), [labels[i % len(labels)] for i in range(40)]
    joblib.dump({"model": LogisticRegression(max_iter=500).fit(X, y), "labels": labels, "kind": "lung",
                 "threshold": None, "positive": None, "trained_on": "icbhi"}, d / "lung_sound_ast.head")
    (d / "lung_sound_ast.onnx.json").write_text(json.dumps(
        {"role": "embedder", "for": "lung_sound", "head": "lung_sound_ast.head"}))
    return d


def test_ast_embedder_head_replaces_joblib(models_dir, tmp_path):
    labels = ["both", "crackle", "normal", "wheeze"]
    a = Analyzer(_ast_models(models_dir, tmp_path, labels))
    assert "lung_sound_ast" not in a.heads
    r = a.analyze(wav_bytes(tone(600, 10)), "lung")
    h = r["heads"]["lung_sound"]
    assert set(h["probs"]) == set(labels) and h["trained_on"] == "icbhi"
    assert r["window_scores"]


def test_ast_head_scores_at_most_twelve_windows(models_dir, tmp_path):
    a = Analyzer(_ast_models(models_dir, tmp_path, ["crackle", "normal"]))
    r = a.analyze(wav_bytes(tone(600, 60)), "lung")
    s = r["window_scores"]
    assert 2 <= len(s) <= 12
    assert s[-1]["end"] <= 60 + 1e-6
