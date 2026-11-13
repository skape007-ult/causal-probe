"""Activation cache: one forward pass per item, saved to disk in float16.

Layout:  cache/<model>[-rand]/<tag>/
    chunk_0000.npz ...   resumable work units (deleted after consolidation)
    resid.npy            float16 [n_items, n_layers+1, n_positions, d_model]
                         index 0 = input to block 0 (embeddings), index l+1 = output of block l
    meta.json            item uids (same order as get_dataset(tag)), positions, answer logits
    readout.npz          final-norm parameters + unembedding rows of the answer tokens,
                         so the logit lens can run from the cache on CPU without the model
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data, models, prompts
from .config import CACHE, K_SHOTS, POSITIONS, SEED

# Tokens per batch (batch size x longest item). Gemma runs one item at a time: on this
# machine batching gives it no speed-up (the GPU is already saturated) and the extra
# activation memory pushes it into swap (measured: B=1 0.6 s/item, B=4 2.6-6 s/batch).
TOKEN_BUDGET = {"qwen0.5b": 6000, "qwen1.5b": 3500, "gemma2b": 1}
# On a CUDA GPU with ~15 GB (Kaggle T4 / P100) every model here can take a real batch.
CUDA_TOKEN_BUDGET = {"qwen0.5b": 16000, "qwen1.5b": 12000, "gemma2b": 6000, "qwen3b": 6000, "qwen7b-4bit": 4000}


def token_budget(model_name: str, device=None) -> int:
    """Tokens per batch. CAUSAL_PROBE_TOKEN_BUDGET overrides (e.g. to recover from an OOM)."""
    import os
    if os.environ.get("CAUSAL_PROBE_TOKEN_BUDGET"):
        return int(os.environ["CAUSAL_PROBE_TOKEN_BUDGET"])
    if device is not None and str(device).startswith("cuda"):
        return CUDA_TOKEN_BUDGET.get(model_name, 4000)
    return TOKEN_BUDGET.get(model_name, 2000)


def cache_dir(model_name: str, tag: str, random_init: bool = False) -> Path:
    return CACHE / (model_name + ("-rand" if random_init else "")) / tag


def fewshot_examples(dataset: str) -> list[dict]:
    """Fixed demonstrations. Corr2Cause: from dev (short premises, <=4 variables).
    CLadder: from the full set minus our subsample."""
    if dataset == "c2c":
        test_prems = {i["group"] for i in data.load_c2c("test")}
        pool = [i for i in data.load_c2c("dev") if i["group"] not in test_prems]
        return prompts.pick_shots(pool, K_SHOTS, SEED, max_chars=450)
    used = {i["uid"] for i in data.cladder_sample()}
    pool = [i for i in data.load_cladder() if i["uid"] not in used]
    return prompts.pick_shots(pool, K_SHOTS, SEED, max_chars=450)


def make_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Length-sorted batches with (batch size x longest item) <= token budget."""
    order = np.argsort(lengths, kind="stable")
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > budget:
            batches.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        batches.append(cur)
    return batches


class ResidCollector:
    """Forward hooks that keep, for every layer, only the residual vectors at the
    requested (batch, position) indices - never the full [B, T, d] tensors."""

    def __init__(self, model):
        self.layers = models.get_layers(model)
        self.idx = None  # LongTensor [B, P] of absolute (padded) positions
        self.out: list = []
        self.handles = [self.layers[0].register_forward_pre_hook(self._pre)]
        self.handles += [l.register_forward_hook(self._post) for l in self.layers]

    def _take(self, h):
        b = torch.arange(h.shape[0], device=h.device)[:, None]
        self.out.append(h[b, self.idx].float().cpu())

    def _pre(self, module, args, kwargs=None):
        self._take(args[0] if args else kwargs["hidden_states"])

    def _post(self, module, args, output):
        self._take(output[0] if isinstance(output, tuple) else output)

    def pop(self) -> torch.Tensor:
        r = torch.stack(self.out, 1)  # [B, L+1, P, d]
        self.out = []
        return r

    def remove(self):
        for h in self.handles:
            h.remove()


def save_readout(model, ans: dict, path: Path):
    norm = model.model.norm
    ids = list(ans["yes"].values()) + list(ans["no"].values())
    np.savez(path,
             norm_weight=norm.weight.detach().float().cpu().numpy(),
             norm_eps=float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))),
             norm_type=type(norm).__name__,
             W_U=models.unembed_rows(model, ids).numpy(),
             n_yes=len(ans["yes"]),
             softcap=float(getattr(model.config, "final_logit_softcapping", 0) or 0))


@torch.no_grad()
def build_cache(model_name: str, tag: str, random_init: bool = False, model=None, tok=None,
                limit: int | None = None, chunk_size: int = 256, log=print) -> Path:
    """Run every item of `tag` through the model once; write chunks; consolidate."""
    out = cache_dir(model_name, tag, random_init)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        log(f"[cache] {out} already complete")
        return out
    fewshot = tag.endswith("_fewshot")
    items = data.get_dataset(tag.removesuffix("_fewshot"))
    if limit:
        items = items[:limit]
    shots = fewshot_examples(items[0]["dataset"]) if fewshot else None
    tok = tok or models.load_tokenizer(model_name)
    model = model or models.load_model(model_name, random_init=random_init)
    device = next(model.parameters()).device
    ans = models.answer_token_ids(tok)
    yes_ids, no_ids = list(ans["yes"].values()), list(ans["no"].values())
    save_readout(model, ans, out / "readout.npz")
    # With the embedding matrix on disk (Gemma), full-vocab logits per batch would page 1.2 GB
    # through memory and stall the GPU; there we store answer-token logits only.
    full_vocab = models.has_cheap_full_vocab(model)

    encs = [prompts.encode_with_positions(tok, it, shots) for it in items]
    lengths = [len(e["input_ids"]) for e in encs]
    chunks = [list(range(s, min(s + chunk_size, len(items)))) for s in range(0, len(items), chunk_size)]
    coll = ResidCollector(model)
    t0, done = time.time(), 0
    for ci, chunk in enumerate(chunks):
        cpath = out / f"chunk_{ci:04d}.npz"
        if cpath.exists():
            continue
        resid = [None] * len(chunk)
        ans_logits = np.zeros((len(chunk), len(yes_ids) + len(no_ids)), np.float32)
        top1 = np.zeros((len(chunk), 2), np.float32)
        for batch in make_batches([lengths[i] for i in chunk], token_budget(model_name, device)):
            gi = [chunk[j] for j in batch]
            T = max(lengths[i] for i in gi)
            ids = torch.full((len(gi), T), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(gi), T), dtype=torch.long)
            idx = torch.zeros((len(gi), len(POSITIONS)), dtype=torch.long)
            for r, i in enumerate(gi):
                n = lengths[i]
                ids[r, T - n:] = torch.tensor(encs[i]["input_ids"])   # left padding
                mask[r, T - n:] = 1
                idx[r] = torch.tensor([encs[i]["pos"][p] + (T - n) for p in POSITIONS])
            coll.idx = idx.to(device)
            h = models.body_forward(model, ids, mask).last_hidden_state[:, -1]
            res = coll.pop()
            al = models.answer_logits_from_hidden(model, h, ans)
            if device.type == "mps" and not full_vocab:
                torch.mps.empty_cache()     # keep the allocator from growing into swap
            if full_vocab:      # top-1 token over the whole vocabulary (diagnostic)
                top = models.logits_from_hidden(model, h).max(-1)
            for r, j in enumerate(batch):
                resid[j] = res[r]
                ans_logits[j] = al[r].numpy()
                top1[j] = (top.indices[r].item(), top.values[r].item()) if full_vocab else (-1, np.nan)
        resid = torch.stack(resid).numpy()
        assert np.isfinite(resid).all(), "non-finite activations"
        r16 = resid.astype(np.float16)
        assert np.isfinite(r16).all(), "float16 overflow in cached activations"
        tmp = out / f"tmp_{ci:04d}.npz"
        np.savez(tmp, resid=r16, ans_logits=ans_logits, top1=top1,
                 max_abs=np.abs(resid).max())
        tmp.rename(cpath)
        done += len(chunk)
        log(f"[cache] {model_name}{'-rand' if random_init else ''}/{tag} chunk {ci + 1}/{len(chunks)} "
            f"{done / (time.time() - t0):.2f} items/s")
    coll.remove()

    # consolidate
    parts = [np.load(out / f"chunk_{ci:04d}.npz") for ci in range(len(chunks))]
    resid = np.concatenate([p["resid"] for p in parts])
    np.save(out / "resid.npy", resid)
    ans_logits = np.concatenate([p["ans_logits"] for p in parts])
    top1 = np.concatenate([p["top1"] for p in parts])
    ny = len(yes_ids)
    ld = (torch.logsumexp(torch.tensor(ans_logits[:, :ny]), -1)
          - torch.logsumexp(torch.tensor(ans_logits[:, ny:]), -1)).numpy()
    meta = dict(
        model=model_name, random_init=random_init, tag=tag, n=len(items),
        positions=list(POSITIONS), shape=list(resid.shape), dtype="float16",
        uids=[i["uid"] for i in items], labels=[i["label"] for i in items],
        n_tokens=lengths, pos=[[e["pos"][p] for p in POSITIONS] for e in encs],
        answer_tokens=ans, ans_logits=ans_logits.tolist(), logit_diff=ld.tolist(),
        top1_id=top1[:, 0].astype(int).tolist() if full_vocab else None,
        top1_logit=top1[:, 1].tolist() if full_vocab else None,
        top1_tok=[tok.decode([int(t)]) for t in top1[:, 0]] if full_vocab else None,
        max_abs_activation=float(max(float(p["max_abs"]) for p in parts)),
        fewshot_uids=[s["uid"] for s in shots] if shots else None,
        device=str(device), model_dtype=str(next(model.parameters()).dtype),
    )
    (out / "meta.json").write_text(json.dumps(meta))
    for ci in range(len(chunks)):
        (out / f"chunk_{ci:04d}.npz").unlink()
    return out


def load_cache(model_name: str, tag: str, random_init: bool = False, mmap: bool = True):
    """Returns (resid [n, L+1, P, d] float16 memmap, meta dict, items list)."""
    d = cache_dir(model_name, tag, random_init)
    meta = json.loads((d / "meta.json").read_text())
    resid = np.load(d / "resid.npy", mmap_mode="r" if mmap else None)
    items = data.get_dataset(tag.removesuffix("_fewshot"))[: meta["n"]]
    assert [i["uid"] for i in items] == meta["uids"], "cache/item order mismatch"
    return resid, meta, items


@torch.no_grad()
def generate_answers(model, tok, items: list[dict], shots=None, max_new_tokens: int = 6,
                     budget: int = 2000) -> list[str]:
    """Greedy-decode a short answer for each item (for 'generated text' scoring).
    Own greedy loop with a key-value cache (equivalent to generate(do_sample=False)), so it
    also works when the embedding matrix lives on disk."""
    encs = [tok(prompts.render(tok, it, shots), add_special_tokens=False).input_ids for it in items]
    lengths = [len(e) for e in encs]
    out = [None] * len(items)
    for batch in make_batches(lengths, budget):
        T = max(lengths[i] for i in batch)
        ids = torch.full((len(batch), T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        for r, i in enumerate(batch):
            ids[r, T - lengths[i]:] = torch.tensor(encs[i]); mask[r, T - lengths[i]:] = 1
        new, past, cur = [], None, ids
        for _ in range(max_new_tokens):
            o = models.body_forward(model, cur, mask, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = models.logits_from_hidden(model, o.last_hidden_state[:, -1]).argmax(-1)
            new.append(nxt)
            cur = nxt[:, None]
            mask = torch.cat([mask, torch.ones((len(batch), 1), dtype=torch.long)], 1)
        new = torch.stack(new, 1)
        for r, i in enumerate(batch):
            toks = new[r].tolist()
            for stop in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                         tok.convert_tokens_to_ids("<end_of_turn>")):
                if stop is not None and stop in toks:
                    toks = toks[: toks.index(stop)]
            out[i] = tok.decode(toks, skip_special_tokens=True)
    return out


def parse_answer(text: str) -> int | None:
    """Map generated text to 1 (yes) / 0 (no) / None (neither)."""
    t = text.strip().lower().lstrip("*\"'( ")
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def export_cache(model_name: str, tag: str, dest: Path, random_init: bool = False,
                 positions: tuple[str, ...] = ("final",), with_resid: bool = True) -> int:
    """Copy one cache directory to `dest/<model>[-rand]/<tag>/`, keeping only the listed token
    positions of the residual stream (float16), so it is small enough to download. meta.json is
    rewritten to describe the reduced array; readout.npz, gen.json and steering*.npz are copied.
    `load_cache` reads the result unchanged. Returns the number of bytes written."""
    import shutil
    src = cache_dir(model_name, tag, random_init)
    out = Path(dest) / src.parent.name / tag
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    if (src / "meta.json").exists():
        meta = json.loads((src / "meta.json").read_text())
        keep = [meta["positions"].index(p) for p in positions]
        if with_resid:
            r = np.load(src / "resid.npy", mmap_mode="r")
            np.save(out / "resid.npy", np.ascontiguousarray(r[:, :, keep, :]))
            meta["shape"] = [meta["shape"][0], meta["shape"][1], len(keep), meta["shape"][3]]
        meta["positions_full_cache"] = meta["positions"]
        meta["positions"] = list(positions)
        meta["pos"] = [[row[k] for k in keep] for row in meta["pos"]]
        meta["resid_exported"] = bool(with_resid)
        (out / "meta.json").write_text(json.dumps(meta))
    for f in src.glob("*"):
        if f.name in ("readout.npz", "gen.json") or f.name.startswith("steering"):
            shutil.copy2(f, out / f.name)
    for f in out.glob("*"):
        n += f.stat().st_size
    return n
