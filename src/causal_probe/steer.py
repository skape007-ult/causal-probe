"""Steering: add a vector to the residual stream at one prompt position and read the answer.

Efficiency trick. An intervention at position p cannot change anything computed at earlier
positions (attention is causal). So we run the tokens before p once, keep the key-value cache,
and then run only the tokens from p onwards once per steering vector, as a batch. For the final
position that is a single token; for the end-of-hypothesis position it is the few template tokens
that follow the user's text. `tests/test_steer.py` checks this equals a full forward pass with the
same hook, at both positions.

Layer convention (same as the cache): index 0 = embeddings, index l >= 1 = output of block l-1.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from . import models


class AddAtLast:
    """Hook that adds `vecs` [B, d] to the residual stream at one position (default: the last),
    at cache layer index `layer`. `index` is an index into the sequence of the forward pass the
    hook sees (so 0 when only the tokens from the steered position onwards are run)."""

    def __init__(self, model, layer: int, index: int = -1):
        layers = models.get_layers(model)
        self.vecs = None
        self.index = index
        if layer == 0:
            self.h = layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)
        else:
            self.h = layers[layer - 1].register_forward_hook(self._post)

    def _pre(self, module, args, kwargs):
        if self.vecs is None:
            return None
        h = args[0].clone()
        h[:, self.index] += self.vecs.to(h.device, h.dtype)
        return (h,) + tuple(args[1:]), kwargs

    def _post(self, module, args, output):
        if self.vecs is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, self.index] += self.vecs.to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        self.h.remove()


@torch.no_grad()
def steered_logit_diffs(model, input_ids: list[int], layer: int, vecs: torch.Tensor, ans: dict,
                        sub_batch: int = 16, pos: int | None = None) -> np.ndarray:
    """Answer logit difference for each steering vector in `vecs` [K, d], added at token index
    `pos` (default: the final position) at `layer`. Pass a zero vector to get the unsteered
    baseline. The answer is always read at the final position."""
    dev = next(model.parameters()).device
    ids = torch.tensor([input_ids])
    T = ids.shape[1]
    pos = T - 1 if pos is None else int(pos)
    assert 1 <= pos <= T - 1, (pos, T)
    pre = models.body_forward(model, ids[:, :pos], torch.ones(1, pos, dtype=torch.long), use_cache=True)
    hook = AddAtLast(model, layer, index=0)          # the steered token is the first one of the second pass
    out = []
    try:
        for s in range(0, len(vecs), sub_batch):
            v = vecs[s:s + sub_batch].to(dev)
            k = len(v)
            past = copy.deepcopy(pre.past_key_values)
            past.batch_repeat_interleave(k)
            hook.vecs = v
            o = models.body_forward(model, ids[:, pos:].repeat(k, 1), torch.ones(k, T, dtype=torch.long),
                                    past_key_values=past, use_cache=True)
            al = models.answer_logits_from_hidden(model, o.last_hidden_state[:, -1], ans)
            ny = len(ans["yes"])
            out.append((torch.logsumexp(al[:, :ny], -1) - torch.logsumexp(al[:, ny:], -1)).numpy())
            del past, o
    finally:
        hook.remove()
    return np.concatenate(out)


@torch.no_grad()
def steered_logit_diffs_full(model, input_ids: list[int], layer: int, vecs: torch.Tensor, ans: dict,
                             pos: int | None = None) -> np.ndarray:
    """Reference implementation without the KV-cache trick (one full forward per vector)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([input_ids])
    hook = AddAtLast(model, layer, index=-1 if pos is None else int(pos))
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


def choose_site(model_name: str, tag: str, resid, meta: dict, items: list[dict], position: str = "final",
                stride: int = 1, log=print) -> tuple[int, float, str]:
    """(layer, C, how) for the gold probe at a token position, by the same rule everywhere:
    final token: the layer that 05_confound.py selected if it was run for this model/tag, else the
    quick selection of 07_steering.py (results/steer_select_<model>_<tag>.json);
    other positions: quick selection at that position (results/steer_select_<model>_<tag>_<position>.json).
    Selection uses the selection half of premises/blocks only."""
    import json

    from . import data, probes
    from .config import RESULTS
    if position == "final":
        conf = RESULTS / f"confound_{model_name}_{tag}.json"
        if conf.exists():
            c = json.loads(conf.read_text())
            return int(c["selected_layer_gold"]), float(c.get("selected_C_gold", probes.PROBE_C)), "05_confound"
    path = RESULTS / (f"steer_select_{model_name}_{tag}" + ("" if position == "final" else f"_{position}") + ".json")
    if not path.exists():
        y = np.array(meta["labels"])
        l, c, grid = probes.quick_select(resid, y, items, data.selection_mask(items), stride, log,
                                         pos=meta["positions"].index(position))
        path.write_text(json.dumps({"model": model_name, "tag": tag, "position": position, "selected_layer_gold": l,
                                    "selected_C_gold": c, "stride": stride, "selection_half_cv_accuracy": grid}, indent=1))
    c = json.loads(path.read_text())
    return int(c["selected_layer_gold"]), float(c["selected_C_gold"]), "quick_select"
