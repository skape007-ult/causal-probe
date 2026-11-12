"""Slow tests (need qwen0.5b weights): cache correctness on a handful of items."""
import numpy as np
import pytest
import torch

from causal_probe import cache, data, lens, models, prompts

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tok = models.load_tokenizer("qwen0.5b")
    model = models.load_model("qwen0.5b", device="cpu", dtype=torch.float32)
    import causal_probe.cache as c
    c.CACHE = tmp_path_factory.mktemp("cache")
    out = cache.build_cache("qwen0.5b", "c2c_test", model=model, tok=tok, limit=12, chunk_size=5)
    return tok, model, out


def test_cache_matches_unbatched_forward(setup):
    tok, model, out = setup
    resid, meta, items = cache.load_cache("qwen0.5b", "c2c_test")
    assert resid.shape[:3] == (12, model.config.num_hidden_layers + 1, 3) and resid.dtype == np.float16
    for k in (0, 7, 11):
        enc = prompts.encode_with_positions(tok, items[k])
        with torch.no_grad():
            o = model(torch.tensor([enc["input_ids"]]), output_hidden_states=True)
        for pi, p in enumerate(meta["positions"]):
            ref = torch.stack([h[0, enc["pos"][p]] for h in o.hidden_states]).numpy()
            got = resid[k, :, pi].astype(np.float32)
            # layers 0..L-1: HF hidden_states == our hooks (pre-norm residual stream)
            assert np.allclose(got[:-1], ref[:-1], rtol=2e-2, atol=2e-2), (k, p, np.abs(got - ref).max())
            # last entry: transformers 5.x returns hidden_states[-1] AFTER the final norm
            # (upstream note U6); our cache stores the pre-norm block output.
            with torch.no_grad():
                normed = model.model.norm(torch.tensor(got[-1])).numpy()
            assert np.allclose(normed, ref[-1], rtol=3e-2, atol=3e-2)
        ld = models.answer_logit_diff(o.logits[0, -1], meta["answer_tokens"]).item()
        assert abs(ld - meta["logit_diff"][k]) < 1e-2


def test_lens_last_layer_reproduces_model_logits(setup):
    _, _, out = setup
    resid, meta, _ = cache.load_cache("qwen0.5b", "c2c_test")
    ro = lens.load_readout(out / "readout.npz")
    ld = lens.lens_by_layer(resid[:, :, 0, :], ro)
    assert np.abs(ld[:, -1] - np.array(meta["logit_diff"])).max() < 0.05


def test_generation_parse():
    assert cache.parse_answer(" Yes.") == 1 and cache.parse_answer("no, because") == 0
    assert cache.parse_answer("**No**") == 0 and cache.parse_answer("The answer") is None


def test_greedy_generation_matches_hf_generate(setup):
    tok, model, _ = setup
    items = data.load_c2c("test")[:3] + data.cladder_sample()[:2]
    mine = cache.generate_answers(model, tok, items, max_new_tokens=5)
    for it, m in zip(items, mine):
        enc = tok(prompts.render(tok, it), add_special_tokens=False, return_tensors="pt")
        g = model.generate(**enc, max_new_tokens=5, do_sample=False)
        assert tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True) == m
