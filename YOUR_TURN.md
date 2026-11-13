# Your turn: two small experiments

Neither is implemented anywhere in this repo. Each should take 1–2 hours and runs on the laptop. Both reuse helpers that already exist and are tested, and each comes with checks that tell you whether your implementation is right before you look at the scientific answer.

Setup for both: `source .venv/bin/activate` (or prefix commands with `.venv/bin/python`), and make sure the cache exists (`ls cache/qwen0.5b`). If it does not, run `scripts/build_cache.sh` with `MODELS=qwen0.5b` (about 35 minutes).

---

## Experiment 1 — Does a "correct answer" direction transfer between benchmarks? (CPU only)

**Goal.** The report shows that a linear probe can read the gold label from the residual stream on Corr2Cause, but not much better than a bag-of-words baseline. A sharper test of whether the probe has found anything like "this causal claim is valid" is transfer: train the probe on one benchmark and test it on the other. A direction that encodes validity in general should transfer; one that encodes Corr2Cause surface statistics should fall to chance.

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

**Then** call it for every layer of `qwen0.5b` with `("c2c_probe" → "cladder")` and `("cladder" → "c2c_probe")` and plot accuracy and AUROC against layer.

**How to check your implementation (before interpreting anything).**
1. *Same-dataset sanity.* Temporarily replace the test set by a held-out half of the source premises (use `data.grouped_folds(items, 2, seed=0)`). At the layer and C listed for `qwen0.5b` / Corr2Cause in `report/tables/probes_summary.md` you should land within about 3 points of the "Gold probe acc." there.
2. *Shuffled labels.* Shuffle `ytr` before fitting. Transfer accuracy must be 0.50 ± 0.03 and AUROC 0.50 ± 0.03 at every layer.
3. *Layer 0.* At layer 0 the final-position vector is the embedding of the same template token for every prompt. Accuracy must be exactly the majority rate (0.50) and all scores identical.

**What to report.** Transfer accuracy and AUROC by layer with CIs; whether using target-set standardisation changes anything; whether AUROC is above 0.5 while accuracy is not (which would mean the direction transfers but the threshold does not). State your expectation before you run it.

---

## Experiment 2 — At which layer does steering start to work? (GPU, ~20 minutes of compute)

**Goal.** Tier 2 steers at one layer only (the probe's best layer). The patching results say the answer-relevant information reaches the final token at a specific depth (see `report/tables/patching_summary.md`, column "First layer with ≥ 0.5 recovery at final token"). If steering along the difference-of-means direction works by writing into the same channel the model reads, steering *before* that depth should be weak or erratic and steering *at or after* it should be strong. Test that.

**Write this function** (suggested file: `scripts/yt2_steer_sweep.py`):

```python
def steering_layer_sweep(model_name: str, tag: str, layers: list[int], alpha: float = 2.0,
                         n_items: int = 100, seed: int = 0) -> dict[int, dict]:
    """For each layer: fit the gold difference-of-means direction on the selection half at that
    layer, add +alpha * s_unit * direction at the final token for n_items report-half items, and
    return {layer: {"flip_rate": float, "mean_delta_logit_diff": float, "s_unit": float}}."""
```

**Helpers to use.**
- `cache.load_cache`, `probes.layer_features(resid, layer, 0)` for the activations at each layer.
- `steer.diff_of_means(X, y)` for the unit direction; compute `s_unit` exactly as `scripts/07_steering.py` does (RMS distance from the mean, selection half only), and copy its selection/report split (`np.random.default_rng(SEED + 7)` over sorted unique groups) so your items match.
- `models.load_tokenizer`, `models.load_model`, `models.answer_token_ids`.
- `prompts.encode_with_positions(tok, item)["input_ids"]` and `steer.steered_logit_diffs(model, ids, layer, vecs, ans)`; pass a zero vector as the first row of `vecs` to get the unsteered baseline in the same call.

**How to check your implementation.**
1. *Zero vector.* The baseline logit difference from the zero vector must match `meta["logit_diff"]` for the same item to within 0.1 (float16 noise).
2. *Reproduce Tier 2.* At the layer stored in `results/steering_qwen0.5b_c2c_probe.json` (`"layer"`) with `alpha = 2.0`, your flip rate must fall inside the CI of `directions.gold_diffmeans.add[alpha == 2.0].flip_rate_plus` in that file (use the same 300 items, or expect a slightly wider spread with 100).
3. *Sign.* With `-alpha` the mean change in logit difference must have the opposite sign at every layer where the effect is non-zero.
4. *Random direction.* Replace the direction by `steer.unit(rng.standard_normal(d))`; the flip rate should be far lower than for the difference-of-means direction at the layers where steering works.

**What to report.** Flip rate and mean Δ logit difference against layer, with the patching "arrival" layer marked. Does steering efficacy switch on at the same depth? Is it still effective at the very last layer (where there is no computation left to amplify it)? State your expectation first.

---

When you are done, add a dated entry to `notes/log.md` in the same format as the others: expectation, what would surprise you, result, comparison.
