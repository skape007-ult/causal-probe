"""Slow: the KV-cache steering shortcut must equal a full forward pass with the same hook."""
import numpy as np
import pytest
import torch

from causal_probe import data, models, prompts, steer

pytestmark = pytest.mark.slow


def test_kv_steering_equals_full_forward():
    tok = models.load_tokenizer("qwen0.5b")
    model = models.load_model("qwen0.5b", device="cpu", dtype=torch.float32)
    ans = models.answer_token_ids(tok)
    it = data.load_c2c("test")[40]
    ids = prompts.encode_with_positions(tok, it)["input_ids"]
    g = torch.Generator().manual_seed(0)
    vecs = torch.cat([torch.zeros(1, 896), 3.0 * torch.randn(4, 896, generator=g)])
    for layer in (0, 7, 24):
        fast = steer.steered_logit_diffs(model, ids, layer, vecs, ans, sub_batch=3)
        slow = steer.steered_logit_diffs_full(model, ids, layer, vecs, ans)
        assert np.allclose(fast, slow, atol=2e-3), (layer, fast, slow)
        assert np.abs(fast[1:] - fast[0]).max() > 1e-3 or layer == 24   # steering does something
    base = models.answer_logit_diff(models.final_logits(model, torch.tensor([ids])), ans).item()
    assert abs(fast[0] - base) < 2e-3                                    # zero vector = unsteered
