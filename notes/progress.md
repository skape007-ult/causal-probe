# Progress (read this first after a restart)

Repo root: this directory. Venv: `.venv` (uv, Python 3.12). Run things as `.venv/bin/python scripts/...`.

## Done
- Environment, datasets downloaded to `data/raw/` (gitignored), loaders + tests (28 passing).
- MPS vs CPU verification for qwen0.5b.

## In progress
- MPS verification for qwen1.5b / gemma2b; cache builder (`src/causal_probe/cache.py`).

## Next
- Build caches (scripts/build_cache.sh) → Tier 1 analyses → audit → push by Fri 23:00.
