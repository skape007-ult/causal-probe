# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.
**Rule learned the hard way: one heavy job at a time on this 8 GB machine** (a GPU job plus a multi-process CPU job stall each other). Gemma must run alone.

Last updated: Thu 2026-11-12 23:50 (Toronto).

## Done
- Environment, loaders, prompts, cache builder, tests (fast: `pytest -q tests/test_data.py tests/test_prompts.py`; slow: `tests/test_cache.py tests/test_steer.py`).
- Caches: qwen0.5b, qwen0.5b-rand, qwen1.5b, qwen1.5b-rand (all tags). Gemma: embedding matrix exported to `cache/embed_gemma2b.npy`; nothing else yet.
- Tier 1 analyses (behaviour, lens, probes at C ∈ {0.01, 0.1, 1}, confound incl. fixed-C and structural baseline): qwen0.5b, qwen1.5b.
- Tier 2: qwen0.5b steering + patching; qwen1.5b steering.
- Docs: EXPLAINER (primer T1+T2, decisions D1–D18, figure guide, Tier 1 Q&A draft), log, upstream issues U1–U9, YOUR_TURN.md, table generator.

## In progress
- `scripts/queue_night.sh` (log: `logs/queue.log`): qwen1.5b patching → Gemma cache (trained, random) → Gemma Tier 1 → Gemma steering/patching. Re-runnable; skips finished steps.
- Independent Tier 1 audit (subagent) → `notes/audit_tier1_raw.md`.

## Next
1. Apply audit fixes; re-run `scripts/05_confound.py` for all models (adds `structural_accuracy_all_items`), `09_cross_model.py`, `12_tables.py`, `06_figures.py`; check every figure by eye.
2. Write up Tier 1 (EXPLAINER start-here, report sections, log comparison vs expectations); push. Gate: Fri 23:00.
3. Tier 2 write-up + audit (gate Sat 20:00).
4. Tier 3: `scripts/10_lora_train.py` (then cache `qwen0.5b-lora` + run_tier1), `scripts/11_sae.py --stage features|ablate` (needs Gemma cache). Stop starting new work Sun 09:00.
5. Packaging Sun 10:00–14:00: README, report/report.md, run_all.sh.
