"""Logit lens computed from the cache on CPU (no model needed).

`readout.npz` (written by cache.save_readout) holds the final RMSNorm parameters and the
unembedding rows of the Yes/No answer tokens. Applying them to the residual stream at an
intermediate layer answers: "if the model stopped here, how strongly would it say Yes vs No?"
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def load_readout(path) -> dict:
    r = np.load(path)
    return {k: (r[k].item() if r[k].ndim == 0 else r[k]) for k in r.files}


def apply_final_norm(x: np.ndarray, ro: dict) -> np.ndarray:
    """RMSNorm as implemented by Qwen2 (x/rms * w) and Gemma2 (x/rms * (1 + w))."""
    x = x.astype(np.float32)
    rms = np.sqrt((x ** 2).mean(-1, keepdims=True) + ro["norm_eps"])
    w = ro["norm_weight"]
    if "gemma" in str(ro["norm_type"]).lower():
        w = 1.0 + w
    return x / rms * w


def lens_answer_logits(resid: np.ndarray, ro: dict) -> np.ndarray:
    """resid [..., d] -> logits of the answer tokens [..., n_answer_tokens]."""
    lg = apply_final_norm(resid, ro) @ ro["W_U"].T
    if ro["softcap"]:
        lg = np.tanh(lg / ro["softcap"]) * ro["softcap"]
    return lg


def lens_logit_diff(resid: np.ndarray, ro: dict) -> np.ndarray:
    """resid [..., d] -> logsumexp(Yes variants) - logsumexp(No variants)."""
    lg = lens_answer_logits(resid, ro)
    ny = int(ro["n_yes"])
    return logsumexp(lg[..., :ny], -1) - logsumexp(lg[..., ny:], -1)


def lens_by_layer(resid_final: np.ndarray, ro: dict, chunk: int = 512) -> np.ndarray:
    """resid_final [n, L+1, d] (final prompt position) -> logit diff [n, L+1]."""
    out = [lens_logit_diff(np.asarray(resid_final[i:i + chunk]), ro)
           for i in range(0, resid_final.shape[0], chunk)]
    return np.concatenate(out)
