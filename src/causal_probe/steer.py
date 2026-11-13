"""Steering: add a vector to the residual stream at the final prompt position and read the answer.

Efficiency trick. An intervention at the *final* position cannot change anything computed at
earlier positions (attention is causal). So we run the prompt minus its last token once, keep
the key-value cache, and then run only the last token once per steering vector, as a batch.
`tests/test_steer.py` checks this equals a full forward pass with the same hook.

Layer convention (same as the cache): index 0 = embeddings, index l >= 1 = output of block l-1.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from . import models


class AddAtLast:
    """Hook that adds `vecs` [B, d] to the residual stream at the last position, at cache
    layer index `layer`."""

    def __init__(self, model, layer: int):
        layers = models.get_layers(model)
        self.vecs = None
        if layer == 0:
            self.h = layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)
        else:
            self.h = layers[layer - 1].register_forward_hook(self._post)

    def _pre(self, module, args, kwargs):
        if self.vecs is None:
            return None
        h = args[0].clone()
        h[:, -1] += self.vecs.to(h.dtype)
        return (h,) + tuple(args[1:]), kwargs

    def _post(self, module, args, output):
        if self.vecs is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1] += self.vecs.to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        self.h.remove()


@torch.no_grad()
def steered_logit_diffs(model, input_ids: list[int], layer: int, vecs: torch.Tensor, ans: dict,
                        sub_batch: int = 16) -> np.ndarray:
    """Answer logit difference for each steering vector in `vecs` [K, d] (added at the final
    position at `layer`). Pass a zero vector to get the unsteered baseline."""
    dev = next(model.parameters()).device
    ids = torch.tensor([input_ids])
    T = ids.shape[1]
    pre = models.body_forward(model, ids[:, :-1], torch.ones(1, T - 1, dtype=torch.long), use_cache=True)
    hook = AddAtLast(model, layer)
    out = []
    try:
        for s in range(0, len(vecs), sub_batch):
            v = vecs[s:s + sub_batch].to(dev)
            k = len(v)
            past = copy.deepcopy(pre.past_key_values)
            past.batch_repeat_interleave(k)
            hook.vecs = v
            o = models.body_forward(model, ids[:, -1:].repeat(k, 1), torch.ones(k, T, dtype=torch.long),
                                    past_key_values=past, use_cache=True)
            al = models.answer_logits_from_hidden(model, o.last_hidden_state[:, -1], ans)
            ny = len(ans["yes"])
            out.append((torch.logsumexp(al[:, :ny], -1) - torch.logsumexp(al[:, ny:], -1)).numpy())
            del past, o
    finally:
        hook.remove()
    return np.concatenate(out)


@torch.no_grad()
def steered_logit_diffs_full(model, input_ids: list[int], layer: int, vecs: torch.Tensor, ans: dict) -> np.ndarray:
    """Reference implementation without the KV-cache trick (one full forward per vector)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([input_ids])
    hook = AddAtLast(model, layer)
    out = []
    try:
        for v in vecs:
            hook.vecs = v[None].to(dev)
            h = models.body_forward(model, ids, torch.ones_like(ids)).last_hidden_state[:, -1]
            al = models.answer_logits_from_hidden(model, h, ans)
            ny = len(ans["yes"])
            out.append(float(torch.logsumexp(al[0, :ny], -1) - torch.logsumexp(al[0, ny:], -1)))
    finally:
        hook.remove()
    return np.array(out)


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def probe_direction(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """Unit direction of a standardised logistic-regression probe, mapped back to raw
    activation space (w / sigma), pointing towards class 1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=300).fit(sc.transform(X), y)
    return unit(clf.coef_[0] / sc.scale_)


def diff_of_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return unit(X[y == 1].mean(0) - X[y == 0].mean(0))
