"""Central configuration: paths, seeds, model registry, subsample sizes."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
# The cache and results locations can be redirected (Kaggle runs, tiny local smoke tests) so
# that a test never overwrites the real cache or the committed results.
CACHE = Path(os.environ.get("CAUSAL_PROBE_CACHE", ROOT / "cache"))
RESULTS = Path(os.environ.get("CAUSAL_PROBE_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("CAUSAL_PROBE_FIGURES", ROOT / "figures"))

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
# Models that do not fit this 8 GB laptop; their GPU passes run on Kaggle (KAGGLE.md) and the
# exported final-position cache is analysed locally. Kept out of MODELS so that tests and the
# default loops stay local; ALL_MODELS (used for --model choices) includes them.
REMOTE_MODELS = {
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b-4bit": "Qwen/Qwen2.5-7B-Instruct",     # loaded in 4-bit (bitsandbytes), CUDA only
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",          # float16 spread over two 16 GB GPUs (fallback if 4-bit fails)
}
QUANT_4BIT = {"qwen7b-4bit"}
MULTI_GPU = {"qwen7b"}                              # too large for one 16 GB GPU in float16

# LoRA fine-tuned variants: short name -> (base model short name, adapter directory).
CHECKPOINTS = ROOT / "checkpoints"
LORA = {"qwen0.5b-lora": ("qwen0.5b", CHECKPOINTS / "qwen0.5b-lora" / "final")}
ALL_MODELS = list(MODELS) + list(REMOTE_MODELS) + list(LORA)

# Subsample sizes (see EXPLAINER.md decision log for the arithmetic).
C2C_PROBE_PREMISES = 300      # premises sampled from train
C2C_PROBE_PER_PREMISE = 8     # 4 positive + 4 negative hypotheses per premise
C2C_DEBIASED_BLOCKS = 25      # debiased pool: 25 blocks x 48 hypothesis strings x (1 valid + 1 invalid)
CLADDER_N = 3000              # stratified by label x query_type (rung is a function of query_type)
N_SUB = 500                   # evenly spaced subset per dataset: generated-text scoring and few-shot runs
K_SHOTS = 4

POSITIONS = ("final", "content_end", "premise_end")
