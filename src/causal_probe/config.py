"""Central configuration: paths, seeds, model registry, subsample sizes."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 0
PROBE_SEEDS = (0, 1, 2)
N_FOLDS = 5
N_BOOT = 2000

# short name -> HF id. Order = size order.
MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma2b": "google/gemma-2-2b-it",
}

# Subsample sizes (see EXPLAINER.md decision log for the arithmetic).
C2C_PROBE_PREMISES = 300      # premises sampled from train
C2C_PROBE_PER_PREMISE = 8     # 4 positive + 4 negative hypotheses per premise
CLADDER_N = 3000              # stratified by label x query_type (rung is a function of query_type)
N_GEN = 300                   # items per dataset scored by generated text as well as logits
N_FEWSHOT = 500               # items per dataset for the few-shot behavioural run
K_SHOTS = 4

POSITIONS = ("final", "content_end", "premise_end")
