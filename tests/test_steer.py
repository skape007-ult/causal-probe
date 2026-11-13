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
        assert np.abs(fast[1:] - fast[0]).max() > 1e-3                      # steering changes the answer logits (also at the last layer: the final norm and unembedding still follow)
    base = models.answer_logit_diff(models.final_logits(model, torch.tensor([ids])), ans).item()
    assert abs(fast[0] - base) < 2e-3                                    # zero vector = unsteered


def test_hook_changes_exactly_the_intended_layer():
    """Added after the Tier 2 audit: the test above compares two code paths that share the hook,
    so hooking the wrong layer would pass. Here the residual stream is recorded independently
    (same recorder the cache uses) with and without the hook: at cache index `layer` the final
    position must change by exactly the added vector, earlier indices must not change at all, and
    no other position may change at the hooked index."""
    from causal_probe.cache import ResidCollector
    tok = models.load_tokenizer("qwen0.5b")
    model = models.load_model("qwen0.5b", device="cpu", dtype=torch.float32)
    it = data.load_c2c("test")[40]
    ids = torch.tensor([prompts.encode_with_positions(tok, it)["input_ids"]])
    T = ids.shape[1]
    v = 2.0 * torch.randn(1, 896, generator=torch.Generator().manual_seed(1))

    def record(layer=None):
        # the steering hook is registered first: hooks on a module run in registration order,
        # so the recorder then sees the residual stream after the intervention
        hook = steer.AddAtLast(model, layer) if layer is not None else None
        if hook:
            hook.vecs = v
        coll = ResidCollector(model)
        coll.idx = torch.tensor([[T - 1, T - 2]])          # final position and the one before it
        with torch.no_grad():
            models.body_forward(model, ids, torch.ones_like(ids))
        if hook:
            hook.remove()
        r = coll.pop()[0]                                   # [L+1, 2, d]
        coll.remove()
        return r

    clean = record()
    for layer in (0, 7, 23, 24):
        st = record(layer)
        assert torch.allclose(st[layer, 0] - clean[layer, 0], v[0], atol=1e-4), layer     # exactly +v at the hooked index
        assert torch.allclose(st[:layer], clean[:layer], atol=1e-6), layer                 # nothing earlier changes
        assert torch.allclose(st[layer, 1], clean[layer, 1], atol=1e-6), layer             # other positions untouched there
        if layer < 24:
            assert (st[layer + 1, 0] - clean[layer + 1, 0]).abs().max() > 1e-3             # and it propagates
