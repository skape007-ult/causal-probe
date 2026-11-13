# Your turn: two small experiments

Neither is implemented anywhere in this repo. Each should take 1–2 hours and runs on the laptop. Both reuse helpers that already exist and are tested, and each comes with checks that tell you whether your implementation is right before you look at the scientific answer.

Setup for both: `source .venv/bin/activate` (or prefix commands with `.venv/bin/python`), and make sure the cache exists (`ls cache/qwen0.5b`). If it does not, run `scripts/build_cache.sh` with `MODELS=qwen0.5b` (about 35 minutes).

---

## Experiment 1 — Does a "correct answer" direction transfer between benchmarks? (CPU only)

**Goal.** The report shows that a linear probe can read the gold label from the residual stream on the original Corr2Cause pool, but no better than text-only baselines that exploit the wording of the hypothesis, and that a much weaker signal remains on the debiased pool. A sharper test of whether the probe has found anything like "this causal claim is valid" is transfer: train the probe on one benchmark and test it on the other. A direction that encodes validity in general should transfer; one that encodes Corr2Cause surface statistics should fall to chance.

**Write this function** (suggested file: `scripts/yt1_transfer.py`):

```python
def transfer_probe(model: str, src_tag: str, dst_tag: str, layer: int,
                   pos: int = 0, C: float = 0.1) -> dict:
    """Train a gold-label probe on all items of `src_tag` at (layer, pos); evaluate on all
    items of `dst_tag`. Returns {"accuracy": {...CI...}, "auroc": {...CI...}, "n": int}."""
```

**Helpers to use.**
- `causal_probe.cache.load_cache(model, tag)` → `(resid, meta, items)`; labels are `meta["labels"]`.
- `causal_probe.probes.layer_features(resid, layer, pos)` → `[n, d]` float32 matrix.
- `causal_probe.probes.fit_predict(Xtr, ytr, Xte, C=C)` → decision scores for the test items (it standardises using the *training* set's mean and standard deviation; think about whether that is what you want when the test set is a different dataset, and try both).
- `causal_probe.stats.accuracy`, `stats.auroc`, `stats.bootstrap_ci(fn, (y, s), groups)` with `groups = [i["group"] for i in items]`.

**Then** call it for every layer of `qwen0.5b` with three source → target pairs and plot accuracy and AUROC against layer:
- `("c2c_probe" → "c2c_debiased")`: the sharpest one. If the original-pool probe mostly reads *which hypothesis is asked*, it must fall to 0.50 on the debiased pool, where hypothesis identity says nothing about the label. (The two pools are both drawn from the train split and may share premises; remove from the target any item whose `group` occurs in the source before scoring, and report how many you removed.)
- `("c2c_debiased" → "c2c_probe")`: the reverse. A direction learned where shortcuts are absent, tested where they are present.
- `("c2c_probe" → "cladder")` and `("cladder" → "c2c_probe")`: transfer across benchmarks.

**How to check your implementation (before interpreting anything).**
1. *Same-dataset sanity.* Temporarily replace the test set by a held-out half of the source premises (use `data.grouped_folds(items, 2, seed=0)`). At the layer and C listed for `qwen0.5b` / Corr2Cause in `report/tables/probes_summary.md` you should land within about 3 points of the "Gold probe acc." there.
2. *Shuffled labels.* Shuffle `ytr` before fitting. Transfer accuracy must be 0.50 ± 0.03 and AUROC 0.50 ± 0.03 at every layer.
3. *Layer 0.* At layer 0 the final-position vector is the embedding of the same template token for every prompt. Accuracy must be exactly the majority rate (0.50) and all scores identical.

**What to report.** Transfer accuracy and AUROC by layer with CIs for each pair, next to the within-pool numbers from `report/tables/audited_verdict.md`; whether using target-set standardisation changes anything; whether AUROC is above 0.5 while accuracy is not (which would mean the direction transfers but the threshold does not). State your expectation before you run it.

---

## Experiment 2 — At which layer does the answer start to respond to steering? (GPU, ~20 minutes of compute)

**Goal.** Tier 2 steers at one layer (the probe's selected layer) and, for some models, two fixed depths. The Tier 2 audit pointed out that one site cannot say much. Patching says the effect of changing a variable in the prompt reaches the final token at a specific depth (see `report/tables/patching_summary.md`, column "First layer with ≥ 0.5 recovery at final token"). Question: how does the *sensitivity* of the answer to a push along the difference-of-means direction change with depth, and is there a step at the same place?

**The metric (use this, not flip rates).** For one item and one unit direction `d` at one layer:

`slope = (LD(+a) − LD(−a)) / (2a)`, where `LD` is the Yes−No answer logit difference and the vector added at the final token is `±a · s · d`, with `s` the RMS distance of the layer's activations from their mean. Average the slope over items. Flip rates depend on how close the model's answers already are to the threshold and are a poor measure for models that say "No" to almost everything.

**Write this function** (suggested file: `scripts/yt2_steer_sweep.py`):

```python
def steering_layer_sweep(model_name: str, tag: str, layers: list[int], a: float = 1.0,
                         n_items: int = 100, seed: int = 0) -> dict[int, dict]:
    """For each layer: fit the gold difference-of-means direction on the selection half at that
    layer, add +a*s*d and -a*s*d at the final token for n_items report-half items, and return
    {layer: {"slope": float, "slope_ci": (lo, hi), "s_unit": float, "own_sd_over_s": float}}."""
```

**Helpers to use.**
- `cache.load_cache`, `probes.layer_features(resid, layer, 0)` for the activations at each layer.
- `data.selection_mask(items)` for the selection half (True) and report half (False). Fit directions on the selection half only; evaluate on report-half items only.
- `steer.diff_of_means(X, y)` for the unit direction; `s` exactly as `scripts/07_steering.py` computes `s_unit` (selection half).
- `models.load_tokenizer`, `models.load_model`, `models.answer_token_ids`.
- `prompts.encode_with_positions(tok, item)["input_ids"]` and `steer.steered_logit_diffs(model, ids, layer, vecs, ans)`; pass a zero vector as the first row of `vecs` to get the unsteered value in the same call.
- `stats.bootstrap_ci(fn, (values,), groups)` with `groups = [i["group"] for i in items]` for the interval.

**How to check your implementation.**
1. *Zero vector.* The unsteered logit difference must match `meta["logit_diff"]` for the same item to within 0.1 (float16 noise).
2. *Reproduce Tier 2.* At the layer stored in `results/steering_analysis_qwen0.5b_c2c_probe.json` (`"layer"`), your slope for the difference-of-means direction must fall inside the interval of `directions.gold_diffmeans.slope_per_s_at_alpha_1` in that file (use the same 300 items, or expect a slightly wider spread with 100).
3. *Antisymmetry.* Reversing the direction must reverse the sign of the slope exactly.
4. *Linearity.* The slope at `a = 0.5` and at `a = 1` should agree to within about 10% at layers where the effect is not tiny; if not, you are outside the linear range and should say so.
5. *Hook location.* `tests/test_steer.py::test_hook_changes_exactly_the_intended_layer` shows how to verify that the vector lands at the layer you asked for.

**What to report.** Slope against layer with intervals, with the patching "arrival" layer marked; the same in units of the direction's own natural SD (`own_sd_over_s`), since the difference of means changes character across layers. Does sensitivity switch on at the arrival depth, rise gradually, or peak and fall? What happens at the last layer, where only the final normalisation and the unembedding remain? State your expectation first.

---

When you are done, add a dated entry to `notes/log.md` in the same format as the others: expectation, what would surprise you, result, comparison.
