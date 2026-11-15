import os

import numpy as np
import pytest

from ml.run_lung_ast import CYCLE_S, LAYERS, fit_cycle


def test_short_cycle_is_repeat_padded_to_fixed_length():
    seg = np.arange(1000, dtype=np.float32)
    out = fit_cycle(seg, 4000)
    assert len(out) == int(CYCLE_S * 4000)
    assert np.array_equal(out[1000:2000], seg)


def test_long_cycle_is_cut_to_fixed_length():
    out = fit_cycle(np.ones(int(CYCLE_S * 4000) + 500), 4000)
    assert len(out) == int(CYCLE_S * 4000)


@pytest.mark.skipif(os.environ.get("RUN_TORCH_TESTS") != "1", reason="downloads AST weights; set RUN_TORCH_TESTS=1")
def test_slim_ast_keeps_pretrained_weights_and_reads_short_input():
    import torch
    from transformers import ASTModel
    from ml.run_lung_ast import AST_ID, MAX_FRAMES, N_LAYERS, slim_ast
    small, full = slim_ast(), ASTModel.from_pretrained(AST_ID)
    assert small.config.num_hidden_layers == N_LAYERS
    assert torch.equal(small.encoder.layer[0].attention.attention.query.weight, full.encoder.layer[0].attention.attention.query.weight)
    out = small(torch.zeros(1, MAX_FRAMES, 128)).hidden_states
    assert len(out) == N_LAYERS + 1 and out[LAYERS[-1]].shape[-1] == 768


@pytest.mark.skipif(os.environ.get("RUN_TORCH_TESTS") != "1", reason="compares against the torch-based extractor")
def test_numpy_fbank_matches_transformers_extractor():
    from transformers import ASTFeatureExtractor
    from ml.ast_fbank import fbanks
    waves = [np.random.default_rng(i).normal(0, 0.1, 16000 * s).astype(np.float32) for i, s in ((0, 4), (1, 12))]
    ref = ASTFeatureExtractor()(waves, sampling_rate=16000, return_tensors="np")["input_values"]
    assert np.allclose(fbanks(waves), ref, atol=1e-5)
