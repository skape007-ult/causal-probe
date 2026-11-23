"""E6 (PREREG Section 4, descriptive): residual stream of the base model and of every A and B model on POOL-LB, at the
final position and at the last token of the hypothesis, every layer. GPU (Kaggle); --tiny runs on the laptop CPU.

  python finetune/scripts/e6_extract.py --runs base,A_s0,B_s0 --adapters <dir with A_s0/ B_s0/ ...> --out <dir>
  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/e6_extract.py --runs base --tiny --model qwen0.5b --out <dir>

Nothing is trained and nothing is tunable. The forward pass is ft_run.py's scoring pass (same prompt, same left
padding, same float16 autocast, the LoRA adapter applied unmerged through peft), with v1.0's ResidCollector
(src/causal_probe/cache.py, unchanged) taking the residual vectors: index 0 = input to block 0, index l+1 = output of
block l. The answer logit difference is recomputed from the same pass and compared with the model's stored
scores.npz (eval_pool_lb): the file records the largest difference, so a wrong adapter or a different prompt shows.

Outputs in --out/<run>/:
  resid.npy   float16 [n_items, n_layers + 1, 2, d_model], positions (final, content_end)
  meta.json   uids (eval_pool_lb order), labels, token counts, positions, answer logit differences, adapter hash
  DONE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune"), str(ROOT / "finetune" / "scripts")]

import numpy as np          # noqa: E402
import torch                # noqa: E402

from causal_probe import models, prompts           # noqa: E402
from causal_probe.cache import ResidCollector        # noqa: E402
from causal_probe.config import MODELS, REMOTE_MODELS   # noqa: E402
from ftstudy import datasets as S                   # noqa: E402
from ft_run import groups_within, pad_left          # noqa: E402

POSITIONS = ("final", "content_end")
SET = "eval_pool_lb"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def extract(model, tok, ans, items, device, amp, budget, log):
    inner = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    model.eval()
    enc = [prompts.encode_with_positions(tok, it) for it in items]
    lens = [len(e["input_ids"]) for e in enc]
    order = np.argsort(lens)[::-1]
    col = ResidCollector(inner)
    n_layers = len(col.layers)
    d = inner.config.hidden_size
    resid = np.zeros((len(items), n_layers + 1, len(POSITIONS), d), np.float16)
    diff = np.zeros(len(items), np.float32)
    t0, done = time.time(), 0
    try:
        for g, idx in enumerate(groups_within(lens, order, budget)):
            ids, mask = pad_left([enc[i]["input_ids"] for i in idx], tok.pad_token_id, device)
            T = ids.shape[1]
            col.idx = torch.tensor([[T - lens[i] + enc[i]["pos"][p] for p in POSITIONS] for i in idx])
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                h = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1]
                logits = inner.lm_head(h).float()
            r = col.pop()
            if not torch.isfinite(r).all():
                raise RuntimeError("non-finite activations")
            resid[idx] = r.numpy().astype(np.float16)
            diff[idx] = models.answer_logit_diff(logits.cpu(), ans).numpy()
            done += len(idx)
            if g % 40 == 0:
                log(f"    {done}/{len(items)} items, {time.time() - t0:.0f} s")
    finally:
        col.remove()
    return resid, diff, np.array(lens, np.int32), [[e["pos"][p] for p in POSITIONS] for e in enc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated: base, A_s0 ... B_s4")
    ap.add_argument("--adapters", default="", help="folder holding <run>/adapter_model.safetensors (not needed for base)")
    ap.add_argument("--stored-scores", default="", help="folder holding <run>/scores.npz to compare with (optional)")
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-tokens", type=int, default=20000)
    ap.add_argument("--tiny", action="store_true", help="smoke test: 24 shortest items")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    hf_id = {**MODELS, **REMOTE_MODELS}[a.model]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ans = models.answer_token_ids(tok)
    items = S.materialise(SET)
    if a.tiny:
        items = sorted(items, key=lambda i: len(i["context"]))[:24]
    for run in a.runs.split(","):
        out = Path(a.out) / run
        out.mkdir(parents=True, exist_ok=True)

        def log(*x):
            print(f"[e6 {run}] " + " ".join(map(str, x)), flush=True)
        if (out / "DONE").exists():
            log("already done"); continue
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float16 if amp else torch.float32).to(device)
        adapter_sha = None
        if run != "base":
            from peft import PeftModel
            ad = Path(a.adapters) / run
            adapter_sha = sha256(ad / "adapter_model.safetensors")
            model = PeftModel.from_pretrained(model, str(ad)).to(device)
        log(f"{len(items)} items of {SET}, device {device}, adapter sha256 {adapter_sha}")
        t0 = time.time()
        resid, diff, lens, pos = extract(model, tok, ans, items, device, amp, a.eval_tokens, log)
        meta = {"run": run, "set": SET, "model": hf_id, "positions": list(POSITIONS), "shape": list(resid.shape),
                "dtype": "float16", "index": "0 = input to block 0, l+1 = output of block l", "tiny": a.tiny,
                "uids": [i["uid"] for i in items], "labels": [int(i["label"]) for i in items], "n_tokens": lens.tolist(),
                "pos": pos, "answer_logit_diff": diff.tolist(), "adapter_sha256": adapter_sha, "seconds": round(time.time() - t0, 1),
                "versions": {m: __import__(m).__version__ for m in ("torch", "transformers", "peft", "numpy")}}
        sp = Path(a.stored_scores) / run / "scores.npz" if a.stored_scores else None
        if sp and sp.exists() and not a.tiny:
            st = np.load(sp)
            if list(st[SET + "__uid"]) != meta["uids"]:
                raise RuntimeError("stored scores are in another item order")
            meta["max_abs_diff_vs_stored_scores"] = float(np.abs(st[SET + "__score"] - diff).max())
            log("largest difference from the stored answer scores:", meta["max_abs_diff_vs_stored_scores"])
        np.save(out / "resid.npy", resid)
        (out / "meta.json").write_text(json.dumps(meta))
        (out / "DONE").write_text("done\n")
        log(f"done in {meta['seconds']:.0f} s, resid {resid.nbytes / 1e6:.0f} MB")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
