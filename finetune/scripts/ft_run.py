"""Train one model of the fine-tuning study and score it on every evaluation set. Runs on one GPU (Kaggle) or, with
--tiny, on the laptop CPU with a stand-in model. Recipe: PREREG.md (version 2), Section 2; nothing here is tunable.

  python finetune/scripts/ft_run.py --condition B --seed 0 --out /kaggle/working/kaggle_out/work
  python finetune/scripts/ft_run.py --condition base --out ...            # the base model: evaluation only
  python finetune/scripts/ft_run.py --condition A --seed 0 --time-test --out ...   # 150 steps, 300 evaluation items
  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/ft_run.py --condition B --seed 0 --tiny --model qwen0.5b --out <dir>

Stages, each skipped if its marker exists (so a cell that is run again resumes):
  train   LoRA adapter; a checkpoint (adapter + optimiser + step) every 200 steps, resumed from if present
  eval    answer logit difference of every item of every evaluation set  -> scores_<condition>_s<seed>.npz
Outputs in --out:  <run>/adapter/ (final adapter, float16),  <run>/train_log.json,  <run>/scores.npz,  <run>/DONE
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]

import numpy as np          # noqa: E402
import torch                # noqa: E402

from causal_probe import models, prompts          # noqa: E402
from causal_probe.config import MODELS, REMOTE_MODELS   # noqa: E402
from ftstudy import datasets as S                  # noqa: E402

RECIPE = dict(rank=16, alpha=32, dropout=0.05, lr=2e-4, warmup=30, batch=16, epochs=3,
              targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
EVAL_SETS = ["eval_pool_lb", "eval_id48", "eval_id", "eval_pool", "eval_official_test", "eval_id48_renamed",
             "eval_id48_renamed_keep_order", "eval_id_renamed", "eval_id_paraphrased", "eval_pool_lb_renamed",
             "eval_authors_refactorization", "eval_authors_paraphrasing"]
CONDITIONS = ("A", "A48", "B", "C", "Bswap")


def encode(tok, items):
    return [prompts.encode_with_positions(tok, it)["input_ids"] for it in items]


def pad_left(seqs, pad_id, device):
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, T - len(s):] = torch.tensor(s); mask[r, T - len(s):] = 1
    return ids.to(device), mask.to(device)


def final_logits(model, ids, mask, amp):
    """Logits at the last position (the answer position; sequences are left-padded)."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        inner = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
        h = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1]
        return inner.lm_head(h).float()


def groups_within(lengths, order, budget):
    """Split `order` (indices, longest first) into consecutive groups with  len(group) x longest item  <= budget."""
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > budget:
            out.append(cur); cur = []
        cur.append(int(i))
    return out + ([cur] if cur else [])


@torch.no_grad()
def score(model, tok, ans, items, device, amp, budget, log=None):
    """Answer logit difference (v1.0 `answer_logit_diff`) and prompt token count for every item, in the items' order."""
    model.eval()
    enc = encode(tok, items)
    order = np.argsort([len(e) for e in enc])[::-1]            # longest first: an out-of-memory error shows at once
    out = np.zeros(len(items), np.float32)
    t0 = time.time()
    done = 0
    for g, idx in enumerate(groups_within([len(e) for e in enc], order, budget)):
        ids, mask = pad_left([enc[i] for i in idx], tok.pad_token_id, device)
        out[idx] = models.answer_logit_diff(final_logits(model, ids, mask, amp).cpu(), ans).numpy()
        done += len(idx)
        if log and g % 40 == 0:
            log(f"    {done}/{len(items)} items, {time.time() - t0:.0f} s")
    return out, np.array([len(e) for e in enc], np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=CONDITIONS + ("base",))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forward-tokens", type=int, default=2600,
                    help="training: padded tokens per forward pass (items x longest). The 16 items of a step are split into forward "
                         "passes within this budget and their gradients accumulated, which is the same step as one batch of 16. "
                         "Halved automatically after an out-of-memory error. (The first time test ran 4 items per pass and ran "
                         "out of memory on a 15 GB T4: prompts reach 2,500 tokens.)")
    ap.add_argument("--eval-tokens", type=int, default=20000, help="scoring: padded tokens per forward pass")
    ap.add_argument("--time-test", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="smoke test: 64 short training items, 6 steps, 24 items per evaluation set")
    a = ap.parse_args()
    run = f"{a.condition}_s{a.seed}" if a.condition != "base" else "base"
    out = Path(a.out) / run
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(*x):
        line = " ".join(str(v) for v in x)
        print(f"[{run}] {line}", flush=True); logf.write(line + "\n"); logf.flush()
    if (out / "DONE").exists():
        log("already done"); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    hf_id = {**MODELS, **REMOTE_MODELS}[a.model]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ans = models.answer_token_ids(tok)
    yes_id, no_id = ans["yes"]["Yes"], ans["no"]["No"]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    # GPU: frozen base weights in float16 (3 GB instead of 6), LoRA weights in float32 (peft's default), float16 autocast
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float16 if amp else torch.float32).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    info = {"run": run, "condition": a.condition, "seed": a.seed, "model": hf_id, "recipe": RECIPE, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "tiny": a.tiny, "time_test": a.time_test,
            "versions": {m: __import__(m).__version__ for m in ("torch", "transformers", "peft", "numpy")}}

    # ------------------------------------------------------------------ train
    if a.condition != "base":
        from peft import LoraConfig, get_peft_model
        train = S.materialise("train_" + a.condition)
        if a.tiny:
            train = sorted(train, key=lambda i: len(i["context"]))[:64]
        enc = encode(tok, train)
        info["train"] = {"n": len(train), "valid": int(sum(i["label"] for i in train)), "sha256": S.digest(train),
                         "mean_tokens": float(np.mean([len(e) for e in enc])), "max_tokens": int(max(map(len, enc)))}
        torch.manual_seed(a.seed)                                     # LoRA initialisation
        model = get_peft_model(model, LoraConfig(r=RECIPE["rank"], lora_alpha=RECIPE["alpha"], lora_dropout=RECIPE["dropout"],
                                                 task_type="CAUSAL_LM", target_modules=RECIPE["targets"]))
        params = [p for p in model.parameters() if p.requires_grad]
        info["trainable_params"] = int(sum(p.numel() for p in params))
        steps_per_epoch = len(train) // RECIPE["batch"]
        total = steps_per_epoch * RECIPE["epochs"]
        run_steps = 6 if a.tiny else 150 if a.time_test else total
        rng = np.random.default_rng(1000 + a.seed)                    # data order
        order = np.concatenate([rng.permutation(len(train))[:steps_per_epoch * RECIPE["batch"]] for _ in range(RECIPE["epochs"])])
        opt = torch.optim.AdamW(params, lr=RECIPE["lr"], weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / RECIPE["warmup"]) * max(0.0, 1 - s / total))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        step, losses, tlog = 0, [], []
        ck = out / "checkpoint.pt"
        if ck.exists():                                               # resume inside a session
            st = torch.load(ck, map_location=device)
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(model, st["adapter"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            scaler.load_state_dict(st["scaler"]); step, tlog = st["step"], st["log"]
            log(f"resumed at step {step}")
        log(f"training {a.condition} seed {a.seed}: {len(train)} items, {run_steps} of {total} steps, batch {RECIPE['batch']} "
            f"in forward passes of at most {a.forward_tokens} padded tokens, {info['trainable_params']:,} trainable parameters, device {device}")
        lens = [len(e) for e in enc]
        budget, n_oom = a.forward_tokens, 0
        model.train()
        t0 = time.time()
        while step < run_steps:
            idx = order[step * RECIPE["batch"]:(step + 1) * RECIPE["batch"]]
            by_len = sorted(idx, key=lambda i: -lens[i])
            while True:
                try:
                    tot = 0.0
                    for sub in groups_within(lens, by_len, budget):
                        ids, mask = pad_left([enc[i] for i in sub], tok.pad_token_id, device)
                        target = torch.tensor([yes_id if train[i]["label"] == 1 else no_id for i in sub], device=device)
                        loss = torch.nn.functional.cross_entropy(final_logits(model, ids, mask, amp), target, reduction="sum") / len(idx)
                        scaler.scale(loss).backward()
                        tot += float(loss)
                    break
                except torch.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    n_oom += 1
                    if budget <= max(lens) or n_oom > 6:
                        raise
                    budget = max(max(lens), budget // 2)
                    log(f"  out of memory at step {step}: forward passes now at most {budget} padded tokens; the step is redone")
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(tot); step += 1
            if step % 25 == 0 or step == run_steps:
                tlog.append({"step": step, "loss": float(np.mean(losses[-25:])), "minutes": round((time.time() - t0) / 60, 2)})
                log(f"  step {step}/{run_steps} loss {tlog[-1]['loss']:.4f}  {(time.time() - t0) / max(1, len(losses)):.2f} s/step")
            if step % 200 == 0 and step < run_steps:
                from peft import get_peft_model_state_dict
                torch.save({"adapter": get_peft_model_state_dict(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": step, "log": tlog}, ck)
        info["train"].update(steps=step, total_steps=total, log=tlog, seconds_per_step=(time.time() - t0) / max(1, len(losses)),
                             first_loss=tlog[0]["loss"] if tlog else None, last_loss=tlog[-1]["loss"] if tlog else None)
        if device == "cuda":
            info["train"]["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            info["train"].update(forward_tokens_final=budget, out_of_memory_retries=n_oom)
        if not a.time_test:
            model.save_pretrained(out / "adapter")
            from safetensors.torch import load_file, save_file
            f = out / "adapter" / "adapter_model.safetensors"
            save_file({k: v.half() for k, v in load_file(f).items()}, f)       # float16: about 37 MB instead of 74
        if ck.exists():
            ck.unlink()

    # ------------------------------------------------------------------ evaluate
    scores, t0 = {}, time.time()
    for name in EVAL_SETS:
        items = S.materialise(name)
        if a.tiny:
            items = sorted(items, key=lambda i: len(i["context"]))[:24]
        elif a.time_test:
            items = items[:: max(1, len(items) // 25)][:25]
        log(f"  scoring {name}: {len(items)} items")
        s, n = score(model, tok, ans, items, device, amp, a.eval_tokens, log)
        scores[name + "__score"], scores[name + "__n_tokens"] = s, n
        scores[name + "__uid"] = np.array([i["uid"] for i in items]); scores[name + "__label"] = np.array([i["label"] for i in items], np.int8)
        if not np.isfinite(s).all():
            raise RuntimeError(f"non-finite scores on {name}")
    info["eval"] = {"seconds": round(time.time() - t0, 1), "items": int(sum(len(v) for k, v in scores.items() if k.endswith("__score"))),
                    "sets": EVAL_SETS}
    if not a.time_test:                      # the time test's scores are not kept (PREREG Section 2)
        np.savez_compressed(out / "scores.npz", **scores)
    (out / "run.json").write_text(json.dumps(info, indent=1))
    (out / "DONE").write_text("done\n")
    log(f"done: evaluation {info['eval']['items']} items in {info['eval']['seconds']:.0f} s"
        + (f"; training {info['train']['seconds_per_step']:.2f} s/step" if "train" in info else ""))


if __name__ == "__main__":
    main()
