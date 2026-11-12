"""Model loading (plain Hugging Face + hooks), answer-token handling, random-init control."""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import MODELS, SEED

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO", " NO"]


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(MODELS.get(name, name))
    tok.padding_side = "left"  # so position -1 is always the final prompt token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, device: str | None = None, dtype=None, random_init: bool = False):
    """Load an instruct model. `random_init=True` gives an untrained copy with the same
    architecture and tokenizer (the 'random model' probing control)."""
    hf_id = MODELS.get(name, name)
    device = device or pick_device()
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else DEFAULT_DTYPE.get(name, torch.float16)
    if random_init:
        torch.manual_seed(SEED)
        cfg = AutoConfig.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_config(cfg).to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# Filled in after the MPS-vs-CPU verification (scripts/00_verify_env.py); see decision log.
DEFAULT_DTYPE: dict = {}


def answer_token_ids(tok) -> dict:
    """Single-token ids for each Yes/No surface variant ('Yes', ' yes', ...).
    Variants that are not a single token for this tokenizer are dropped."""
    def ids(words):
        out = {}
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                out[w] = t[0]
        return out
    yes, no = ids(YES_VARIANTS), ids(NO_VARIANTS)
    assert yes and no
    return {"yes": yes, "no": no}


def answer_logit_diff(logits: torch.Tensor, ans: dict) -> torch.Tensor:
    """logsumexp over Yes-variants minus logsumexp over No-variants (last dim = vocab)."""
    y = torch.logsumexp(logits[..., list(ans["yes"].values())].float(), -1)
    n = torch.logsumexp(logits[..., list(ans["no"].values())].float(), -1)
    return y - n


def get_layers(model):
    """The list of transformer blocks (Qwen2 and Gemma2 both expose model.model.layers)."""
    return model.model.layers


def unembed(model, resid: torch.Tensor) -> torch.Tensor:
    """Logit lens readout: final norm + unembedding applied to a residual-stream vector."""
    h = model.model.norm(resid)
    logits = model.lm_head(h)
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:  # Gemma-2 soft-caps its final logits
        logits = torch.tanh(logits / cap) * cap
    return logits
