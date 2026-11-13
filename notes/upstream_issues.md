# Upstream issues (not filed; for the owner to verify and file)

Each entry: what, where, minimal reproduction, status of my own verification.

## U1. Corr2Cause and CLadder dataset cards are empty (HF datasets)
- `https://huggingface.co/datasets/causal-nlp/corr2cause/raw/main/README.md` ends with `# Dataset card for corr2cause` / `TODO`. CLadder's card is YAML only. Neither states a licence (`license` tag absent in `https://huggingface.co/api/datasets/causal-nlp/corr2cause`). The GitHub code repos (causalNLP/corr2cause, causalNLP/cladder) are MIT; whether that covers the data is not stated.
- Repro: `curl -sL https://huggingface.co/datasets/causal-nlp/corr2cause/raw/main/README.md | tail -3`

## U2. Corr2Cause dev shares premises with train
- 2 of the 22 distinct dev premises (42 of 1,076 dev items) also occur as premises in train. Test has no premise overlap.
- Repro:
```python
import csv; csv.field_size_limit(10**9)
prem = lambda r: r["input"].split("\nHypothesis: ")[0]
tr = {prem(r) for r in csv.DictReader(open("train.csv"))}
dev = list(csv.DictReader(open("dev.csv")))
print(sum(prem(r) in tr for r in dev), len({prem(r) for r in dev} & tr))   # 42 2
```

## U3. Corr2Cause: `child` template has no positive examples in any split
- `template == "child"`: 34,291 train / 177 dev / 194 test items, all label 0; `parent` is 35% positive. By symmetry one would expect "B directly causes A"-type hypotheses to be valid about as often. May be intentional (e.g. deduplication of symmetric pairs) but is undocumented, and makes the template a strong label shortcut.
- Repro: `collections.Counter((r["template"], r["label"]) for r in csv.DictReader(open("train.csv")))`

## U4. Corr2Cause perturbation files do not align with test.csv
- `perturbation_by_refactorization_test.csv` and `perturbation_by_paraphrasing_test.csv` have 2,246 rows vs 1,162 in `test.csv`; the paraphrasing file's `label` column is 3-class (0/1/2) while the default config is binary; the refactorization file has no `template`/`num_variables` columns. So a paired original-vs-perturbed comparison on the released test set is not possible from these files. (We therefore apply our own A,B,C→Z,Y,X renaming to `test.csv`.)
- Repro: `wc -l test.csv perturbation_by_*_test.csv`; `Counter(r["label"] ...)`.

## U5. TransformerLens 4: `HookedTransformer` import fails with an ImportError that points at a doc path
- `from transformer_lens import HookedTransformer` raises `ImportError: 'HookedTransformer' was removed in TransformerLens 4.0 ... See docs/source/content/migrating_to_v4.md` — a repo-relative path that does not exist in the installed wheel, with no URL. Minor doc gap. (transformer-lens 4.4.0.)

## U6. transformers 5.22: `output_hidden_states` last entry is post-final-norm for Qwen2 (silent trap for logit lens)
- With `Qwen/Qwen2.5-0.5B-Instruct`, `out.hidden_states[-1]` equals `model.model.norm(last_block_output)`, while `hidden_states[:-1]` are raw (pre-norm) block outputs. Anyone doing a logit lens via `hidden_states` and applying `model.model.norm` themselves double-normalises the last layer only. The docstring ("hidden-states of the model at the output of each layer") does not say the final norm is included. I have not checked whether this differs from transformers 4.x or across architectures.
- Repro (prints False / True / True):
```python
import torch; from transformers import AutoModelForCausalLM, AutoTokenizer
n="Qwen/Qwen2.5-0.5B-Instruct"; tok=AutoTokenizer.from_pretrained(n); m=AutoModelForCausalLM.from_pretrained(n)
g={}; m.model.layers[-1].register_forward_hook(lambda mod,a,o: g.__setitem__("x", o[0] if isinstance(o,tuple) else o))
m.model.layers[-2].register_forward_hook(lambda mod,a,o: g.__setitem__("x2", o[0] if isinstance(o,tuple) else o))
hs=m(tok("The cat sat on the mat",return_tensors="pt").input_ids, output_hidden_states=True).hidden_states
print(torch.allclose(hs[-1],g["x"],atol=1e-4), torch.allclose(hs[-1],m.model.norm(g["x"]),atol=1e-4), torch.allclose(hs[-2],g["x2"],atol=1e-4))
```
- Our cache uses forward hooks on the blocks, so it is unaffected (tests/test_cache.py checks both facts).

## U7. SAELens 6.55: `sae.cfg.hook_name` is None for Gemma Scope SAEs (moved to `cfg.metadata`)
- `SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical")` returns an SAE whose `cfg.hook_name` is `None`; the hook name (`blocks.12.hook_resid_post`) now lives in `sae.cfg.metadata.hook_name`. Older tutorials read `sae.cfg.hook_name`. Also `from_pretrained` now returns the SAE itself rather than the `(sae, cfg_dict, sparsity)` tuple many examples unpack. Doc/migration gap rather than a bug.
- Repro: `from sae_lens import SAE; s=SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id="layer_12/width_16k/canonical"); print(type(s), s.cfg.hook_name, s.cfg.metadata.hook_name)`

## U8. transformers 5.22: `model.apply(model._init_weights)` is a silent no-op on a model loaded with `from_pretrained`
- Intending to get a random-init copy without allocating a float32 model, I called `model.apply(model._init_weights)` on a loaded checkpoint. No error, no warning, weights unchanged (modules are flagged as already initialised and skipped). A warning, or a documented `reinitialize()` helper, would prevent a silent "random-init control" that is actually the trained model.
- Repro (prints `True`: weights identical after the call):
```python
import torch; from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
w0 = m.model.layers[3].mlp.up_proj.weight.clone(); m.apply(m._init_weights)
print(torch.equal(w0, m.model.layers[3].mlp.up_proj.weight))
```
- Our workaround: `models.reinit_` (explicit N(0, initializer_range) for Linear/Embedding, constructor defaults for norms); its per-tensor statistics match `from_config` on Qwen2.5-0.5B.

## U9. CLadder: two query types are fully predictable from surface features, and overall ~77% is reachable without causal inference
- A gradient-boosted classifier on shallow features (the percentages quoted in the prompt, two differences between them, polarity words in the question, query type, graph type, and for `backadj` which "Method" is the case-by-case one) reaches ~0.77 accuracy with story-grouped 5-fold CV on a 3,000-item stratified sample of `full_v1.5_default`; `backadj` and `collider_bias` are at 1.00. For `collider_bias` every question of the form "…does it mean that X affects Y?" in the sample has the same answer within a (query type, graph) cell; for `backadj` the answer follows from which method conditions "case by case" and the graph type. This is a benchmark-validity observation rather than a bug: a model can score well on those rungs without reasoning.
- Repro (after `pip install -e .` in this repo and downloading the CSV to `data/raw/cladder/`):
```python
import numpy as np; from causal_probe import data, probes, stats
items = data.get_dataset("cladder"); y = np.array([i["label"] for i in items])
s = probes.structural_oof(items, y, probes.folds_for(items, 0))
qt = np.array([i["query_type"] for i in items])
print(stats.accuracy(y, (s > 0).astype(int)), {t: stats.accuracy(y[qt == t], (s[qt == t] > 0).astype(int)) for t in ("backadj", "collider_bias")})
```
- Not yet checked on the full 10,112 items or against the paper's own shortcut analysis.
