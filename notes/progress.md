# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.
**Rule learned the hard way: one heavy job at a time on this 8 GB machine** (a GPU job plus a multi-process CPU job stall each other). Gemma must run alone.

Last updated: Fri 2026-11-13 09:25 (Toronto), written just before a laptop restart requested by the user. All background jobs were stopped on purpose; nothing is running.

## State at shutdown

### What was running (stopped by hand, Fri 09:22)
- `scripts/queue_night.sh` (started Thu 22:27; log `logs/queue.log`).
- Its current step: `scripts/07_steering.py --model gemma2b --tag c2c_probe` (started Fri 08:27). It printed `0/300 151s` and then nothing for 52 minutes; its resident memory had dropped to about 3 MB, so it was swapped out and stalled, not progressing. It wrote no output and has no checkpointing, so nothing was lost by stopping it. **This step does not work as configured on Gemma** (see "Restart" below).

### What finished overnight (queue steps, all verified to have written their outputs)
- qwen1.5b steering (c2c_probe, cladder) and patching: Thu 23:07 – Fri ~00:00. Already committed.
- Gemma cache, trained model, all 8 tags (`cache/gemma2b/*`): finished Fri 05:41.
- Gemma cache, random-init control (`cache/gemma2b-rand/{c2c_probe,cladder}`): 83.6 min, finished Fri 07:05. The `reinit_` + disk-embedding path ran without error.
- Gemma Tier 1 (`./scripts/run_tier1.sh gemma2b 4`, Fri 07:05 – 08:27): `results/{behaviour,lens,probe_sensitivity}_gemma2b.json`, `results/probes_gemma2b_{c2c_probe,cladder}_C{0.01,0.1,1}.json`, `results/confound_gemma2b_{c2c_probe,cladder}[_fixedC].json`. No errors or tracebacks in `logs/tier1_gemma2b.log`. **These numbers have not been looked at yet, are not in the tables, figures or report, and were not covered by the Tier 1 audit.**

### What did not run
- Gemma steering (c2c_probe: stalled; cladder: never started).
- Gemma patching (`08_patching.py --model gemma2b`): never started.

### Not affected by the restart
- Everything under `cache/`, `data/`, `checkpoints/`, `logs/` is on disk (git-ignored). No cache needs rebuilding. `cache/embed_gemma2b.npy` (Gemma embedding memmap) is in place.
- Hugging Face model downloads are in the usual HF cache.

## Restart (in this order)

1. Sanity check after boot: `.venv/bin/python -m pytest -q tests/test_data.py tests/test_prompts.py tests/test_baselines.py`.
2. CPU only, a few minutes, safe to run immediately — fold Gemma into Tier 1:
   `.venv/bin/python scripts/09_cross_model.py`, then `scripts/12_tables.py`, `scripts/06_figures.py`, `scripts/13_build_docs.py`. Look at every regenerated figure before committing.
3. Gemma steering needs a fix before it is restarted. Do **not** simply re-run `./scripts/queue_night.sh` (it is re-runnable and would skip finished steps, but would walk straight back into the stall). Suspected cause, not yet confirmed: `steer.steered_logit_diffs` batches all steering vectors for an item in one forward pass, which is fine for Qwen but exceeds memory for Gemma (everything else on Gemma runs at batch size 1). Plan: add a `--vec-batch` option (1–2 for Gemma), time 5 items, and only launch the full run if the projected time is under 90 minutes; otherwise cut items (300 → 100) and log the change in `notes/log.md` as a compute-driven change made before seeing any Gemma steering result.
4. Then, one at a time: Gemma steering c2c_probe, Gemma steering cladder, Gemma patching (`--n-pairs 24 --n-comp-pairs 12 --layer-stride 2 --batch 1`). Same timing check first for patching. If Gemma Tier 2 does not fit, report Tier 2 on the two Qwen models and say so.

## Done so far
- Environment, loaders, prompts, cache builder, tests.
- Caches: qwen0.5b, qwen1.5b, gemma2b, each with a random-init control.
- Tier 1 analyses for all three models. For the two Qwen models they are audited, in tables, figures and the report draft, and pushed. Audited verdict for both: the "represents but doesn't say" criterion is not met in any cell; the best text-only baseline beats the probe everywhere.
- Independent Tier 1 audit done (`notes/audit_tier1_raw.md`); fixes applied and logged (decisions D17–D19).
- Tier 2: steering, patching and cross-model comparison for qwen0.5b and qwen1.5b (results committed; write-up and audit pending).
- Docs: EXPLAINER (primer T1+T2, decisions D1–D19, figure guide, Tier 1 Q&A), log, upstream issues U1–U10, YOUR_TURN.md, report template + builder, table generator.

## Next (after the restart steps above)
1. Tier 1 write-up with all three models: report placeholders, EXPLAINER "Start here", log entry comparing results with the logged expectations (including Gemma), light re-check of the Gemma numbers since the audit did not cover them. Push. **Gate: Fri 23:00.**
2. Tier 2: write-up, Tier 2 Q&A, independent audit of `steer.py`, `patch.py`, `07`, `08`, `09` and their results; fix, log, push. **Gate: Sat 20:00.**
3. Tier 3 (only once Tier 2 is pushed; start nothing new after Sun 09:00): `scripts/10_lora_train.py` (untested, about 70 min), cache `qwen0.5b-lora` and run Tier 1 on it; `scripts/11_sae.py --stage features`, then `--stage ablate` (untested draft, needs Gemma alone).
4. Packaging Sun 10:00–14:00: README (needs `report/README_template.md`), report Audits and Limitations sections, `scripts/run_all.sh` (not written yet), finalise YOUR_TURN.md, at least 25 professor questions across tiers.
