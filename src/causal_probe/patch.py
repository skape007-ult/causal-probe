"""Minimal pairs and activation / attribution patching (Corr2Cause).

A minimal pair is (clean, corrupt): two prompts with the SAME number of tokens whose gold
answers differ (clean = Yes, corrupt = No).

  type "hyp"   same premise; the hypothesis differs in one variable letter
               ("A directly causes B." vs "A directly causes C."). Both are dataset items.
  type "prem"  same hypothesis; the premise has two variable letters swapped everywhere, which
               relabels the nodes of the causal graph. By symmetry the gold label of
               (swap(premise), h) equals the dataset label of (premise, swap(h)). The two prompts
               ask the same question about two different causal structures.

Patching copies activations from the clean run into the corrupt run at chosen places and asks
how much of the clean-vs-corrupt difference in the answer logit difference comes back.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import torch

from . import models, prompts

_VARS = re.compile(r"\b([A-F])\b")


def _swap(text: str, a: str, b: str) -> str:
    return _VARS.sub(lambda m: {a: b, b: a}.get(m.group(1), m.group(1)), text)


def build_pairs(items: list[dict], kind: str, n: int, seed: int = 0, max_per_premise: int = 2) -> list[dict]:
    """Find minimal pairs among dataset items (see module docstring)."""
    rng = np.random.default_rng(seed)
    by_prem = defaultdict(dict)
    for it in items:
        by_prem[it["group"]][it["query"]] = it
    pairs = []
    for g in sorted(by_prem):
        hyps = by_prem[g]
        cand = []
        for q, it in hyps.items():
            if it["label"] != 1:
                continue
            vs = sorted(set(_VARS.findall(q)))
            allv = sorted(set(_VARS.findall(it["context"])))
            for a in vs:
                for b in allv:
                    if b == a:
                        continue
                    q2 = _swap(q, a, b)
                    # exactly one letter position changes (b not already in q) and the swapped
                    # hypothesis exists in the data with label 0
                    if b in vs or q2 not in hyps or hyps[q2]["label"] != 0:
                        continue
                    if kind == "hyp":
                        cand.append(dict(kind=kind, clean=it, corrupt=hyps[q2], group=g, swap=(a, b)))
                    else:
                        cor = dict(it)
                        cor["context"] = _swap(it["context"], a, b)
                        cor["label"] = 0
                        cor["uid"] = it["uid"] + f"-swap{a}{b}"
                        cand.append(dict(kind=kind, clean=it, corrupt=cor, group=g, swap=(a, b)))
        if cand:
            for j in rng.choice(len(cand), min(max_per_premise, len(cand)), replace=False):
                pairs.append(cand[int(j)])
    idx = rng.permutation(len(pairs))[:n]
    return [pairs[i] for i in sorted(idx)]


def encode_pair(tok, pair: dict) -> dict | None:
    """Tokenise both prompts; return None if they are not token-aligned.
    Regions (lists of positions) partition the positions from the first difference onwards;
    positions before the first differing token are identical in both runs and are skipped."""
    ec = prompts.encode_with_positions(tok, pair["clean"])
    ek = prompts.encode_with_positions(tok, pair["corrupt"])
    a, b = np.array(ec["input_ids"]), np.array(ek["input_ids"])
    if len(a) != len(b) or ec["pos"] != ek["pos"]:
        return None
    diff = np.where(a != b)[0]
    if len(diff) == 0:
        return None
    pe, ce, T = ec["pos"]["premise_end"], ec["pos"]["content_end"], len(a)
    first = int(diff[0])
    dset = set(diff.tolist())
    regions = {
        "diff_tokens": sorted(dset),
        "premise_rest": [p for p in range(first, pe + 1) if p not in dset],
        "hypothesis": [p for p in range(max(first, pe + 1), ce + 1) if p not in dset],
        "template_tail": list(range(ce + 1, T - 1)),
        "final": [T - 1],
    }
    return dict(clean_ids=a.tolist(), corrupt_ids=b.tolist(), regions=regions, n_tokens=T,
                n_diff=len(diff), pos=ec["pos"])


# ----------------------------------------------------------------------------- hooks
def _out(output):
    return output[0] if isinstance(output, tuple) else output


class ResidRecorder:
    """Record the full residual stream [L+1, T, d] (float16, CPU) for a single prompt."""

    def __init__(self, model):
        self.layers = models.get_layers(model)
        self.buf = []
        self.h = [self.layers[0].register_forward_pre_hook(lambda m, a, k: self.buf.append(a[0][0].to(torch.float16).cpu()), with_kwargs=True)]
        self.h += [l.register_forward_hook(lambda m, a, o: self.buf.append(_out(o)[0].to(torch.float16).cpu())) for l in self.layers]

    def pop(self):
        r = torch.stack(self.buf)
        self.buf = []
        return r

    def remove(self):
        for h in self.h:
            h.remove()


class ResidPatcher:
    """For a batch of B copies of the corrupt prompt, overwrite the residual stream at
    (layer_b, positions_b) with the given clean values, one (layer, region) per batch row."""

    def __init__(self, model):
        self.layers = models.get_layers(model)
        self.jobs = {}  # layer index -> list of (row, positions LongTensor, values [P, d])
        self.h = [self.layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)]
        self.h += [l.register_forward_hook(lambda m, a, o, i=i: self._post(i + 1, o)) for i, l in enumerate(self.layers)]

    def _apply(self, layer, h):
        if layer not in self.jobs:
            return None
        h = h.clone()
        for row, pos, val in self.jobs[layer]:
            h[row, pos] = val.to(h.device, h.dtype)
        return h

    def _pre(self, module, args, kwargs):
        h = self._apply(0, args[0])
        return None if h is None else ((h,) + tuple(args[1:]), kwargs)

    def _post(self, layer, output):
        h = self._apply(layer, _out(output))
        if h is None:
            return None
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        for h in self.h:
            h.remove()


def logit_diff_from_hidden(model, h, ans):
    al = models.answer_logits_from_hidden(model, h, ans)
    ny = len(ans["yes"])
    return (torch.logsumexp(al[:, :ny], -1) - torch.logsumexp(al[:, ny:], -1)).numpy()


@torch.no_grad()
def run_logit_diff(model, ids: list[int], ans, batch: int = 1) -> np.ndarray:
    x = torch.tensor([ids]).repeat(batch, 1)
    h = models.body_forward(model, x, torch.ones_like(x)).last_hidden_state[:, -1]
    return logit_diff_from_hidden(model, h, ans)


@torch.no_grad()
def resid_patch_grid(model, enc: dict, ans, layers: list[int], donor_resid: torch.Tensor,
                     batch: int = 16) -> np.ndarray:
    """Patch donor residuals into the corrupt run for every (layer, region).
    Returns patched logit diffs [len(layers), n_regions]."""
    regions = list(enc["regions"])
    jobs = [(l, r) for l in layers for r in regions if enc["regions"][r]]
    out = np.full((len(layers), len(regions)), np.nan, np.float32)
    patcher = ResidPatcher(model)
    try:
        for s in range(0, len(jobs), batch):
            chunk = jobs[s:s + batch]
            patcher.jobs = defaultdict(list)
            for row, (l, r) in enumerate(chunk):
                pos = torch.tensor(enc["regions"][r])
                patcher.jobs[l].append((row, pos, donor_resid[l, pos]))
            ld = run_logit_diff(model, enc["corrupt_ids"], ans, batch=len(chunk))
            for (l, r), v in zip(chunk, ld):
                out[layers.index(l), regions.index(r)] = v
    finally:
        patcher.remove()
    return out


# ----------------------------------------------------------------------------- components
def n_heads(model) -> int:
    return model.config.num_attention_heads


class ComponentIO:
    """Access to per-head attention outputs (the input of o_proj, split by head) and MLP
    outputs for every layer. Modes:
      record: store activations (optionally with gradients retained)
      patch : replace selected components by donor activations
    """

    def __init__(self, model):
        self.model = model
        self.layers = models.get_layers(model)
        self.H = n_heads(model)
        self.rec = None        # dict name -> tensor when recording
        self.keep_grad = False
        self.patch = {}        # ("z", layer) -> (head list, donor [T, H*dh]) ; ("mlp", layer) -> donor [T, d]
        self.h = []
        for i, l in enumerate(self.layers):
            self.h.append(l.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: self._z(i, a)))
            self.h.append(l.mlp.register_forward_hook(lambda m, a, o, i=i: self._mlp(i, o)))

    def _z(self, i, args):
        z = args[0]
        if ("z", i) in self.patch:
            heads, donor = self.patch[("z", i)]
            dh = z.shape[-1] // self.H
            z = z.clone()
            for hd in heads:
                z[..., hd * dh:(hd + 1) * dh] = donor[..., hd * dh:(hd + 1) * dh].to(z.device, z.dtype)
        if self.rec is not None:
            if self.keep_grad:
                z.retain_grad()
            self.rec[("z", i)] = z
        return (z,)

    def _mlp(self, i, out):
        if ("mlp", i) in self.patch:
            out = self.patch[("mlp", i)].to(out.device, out.dtype)[None].expand_as(out).clone()
        if self.rec is not None:
            if self.keep_grad:
                out.retain_grad()
            self.rec[("mlp", i)] = out
        return out

    def remove(self):
        for h in self.h:
            h.remove()


def record_components(model, io: ComponentIO, ids, ans, grad: bool):
    """Run one prompt; return (logit_diff float, {name: activation [T, ...]}, {name: grad} or None)."""
    io.rec, io.keep_grad, io.patch = {}, grad, {}
    x = torch.tensor([ids])
    with torch.set_grad_enabled(grad):
        if grad:
            # make the graph start at the embeddings so intermediate activations get gradients
            dev = next(model.parameters()).device
            disk = getattr(model, "_disk", None)
            emb = disk.embed(x) if disk is not None else model.model.embed_tokens(x.to(dev))
            emb = emb.detach().requires_grad_(True)
            h = model.model(inputs_embeds=emb).last_hidden_state[:, -1]
            ld = models.answer_logit_diff_from_normed(model, h, ans)[0]
            ld.backward()
            acts = {k: v.detach()[0].float().cpu() for k, v in io.rec.items()}
            grads = {k: v.grad.detach()[0].float().cpu() for k, v in io.rec.items()}
            out = (float(ld.detach()), acts, grads)
        else:
            h = models.body_forward(model, x, torch.ones_like(x)).last_hidden_state[:, -1]
            ld = logit_diff_from_hidden(model, h, ans)[0]
            out = (float(ld), {k: v.detach()[0].float().cpu() for k, v in io.rec.items()}, None)
    io.rec = None
    return out


@torch.no_grad()
def patch_components(model, io: ComponentIO, ids, ans, spec: dict) -> float:
    """spec: {("z", layer): ([heads], donor [T, H*dh]), ("mlp", layer): donor [T, d]}"""
    io.rec, io.patch = None, spec
    x = torch.tensor([ids])
    h = models.body_forward(model, x, torch.ones_like(x)).last_hidden_state[:, -1]
    io.patch = {}
    return float(logit_diff_from_hidden(model, h, ans)[0])
