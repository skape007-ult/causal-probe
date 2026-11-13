"""Tier 3b: LoRA fine-tune the smallest model on Corr2Cause (time-boxed, checkpointed, resumable).

Training data: label-balanced items from train premises that are NOT in the probe pool (and
test premises are disjoint from train anyway), so the Tier 1 analyses of the fine-tuned model
are on premises it never trained on.
Objective: next-token cross-entropy on the answer word ("Yes"/"No") at the final prompt position,
same prompt format as everywhere else.

Writes checkpoints/qwen0.5b-lora/{step_N, final}/ (adapter weights only) and results/lora_training.json.
"""
import argparse
import json
import time

import numpy as np
import torch

from causal_probe import data, models, prompts
from causal_probe.config import CHECKPOINTS, RESULTS, SEED


def make_train_set(n_premises: int, per_premise: int, seed: int):
    pool_groups = {i["group"] for i in data.get_dataset("c2c_probe")}
    train = [i for i in data.load_c2c("train") if i["group"] not in pool_groups]
    return data.c2c_probe_pool(train, seed=seed + 100, n_premises=n_premises, per_premise=per_premise)


@torch.no_grad()
def evaluate(model, tok, items, ans, yes_id, no_id):
    model.eval()
    ld = []
    for it in items:
        ids = torch.tensor([prompts.encode_with_positions(tok, it)["input_ids"]], device=model.device)
        lg = model(input_ids=ids).logits[0, -1].float()
        ld.append(float(models.answer_logit_diff(lg.cpu(), ans)))
    model.train()
    ld = np.array(ld); y = np.array([i["label"] for i in items])
    return {"accuracy": float(((ld > 0) == y).mean()), "frac_yes": float((ld > 0).mean()),
            "mean_logit_diff": float(ld.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="qwen0.5b")
    ap.add_argument("--minutes", type=float, default=70.0)
    ap.add_argument("--n-premises", type=int, default=600)
    ap.add_argument("--per-premise", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(SEED)
    out_dir = CHECKPOINTS / f"{a.base}-lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = models.load_tokenizer(a.base)
    ans = models.answer_token_ids(tok)
    yes_id, no_id = ans["yes"]["Yes"], ans["no"]["No"]
    base = models.load_model(a.base, dtype=torch.float32)       # float32 for stable gradients
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, cfg)
    model.train()
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train = make_train_set(a.n_premises, a.per_premise, SEED)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train))
    encs = {}
    val = [data.get_dataset("c2c_probe")[i] for i in np.linspace(0, 2399, 200).round().astype(int)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = len(train) // a.batch if a.max_steps is None else a.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.1, 1 - s / max(1, total)))

    log_path = RESULTS / "lora_training.json"
    log = {"base": a.base, "rank": a.rank, "lr": a.lr, "batch": a.batch, "n_train_items": len(train),
           "n_train_premises": a.n_premises, "trainable_params": n_train_params, "time_box_minutes": a.minutes,
           "train_label_balance": data.label_balance(train), "steps": [], "eval": []}
    step, t0 = 0, time.time()
    # resume
    ck = sorted(out_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if ck and log_path.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(ck[-1] / "adapter_model.safetensors"))
        step = int(ck[-1].name.split("_")[1])
        log = json.loads(log_path.read_text())
        print(f"resumed from step {step}", flush=True)
    else:
        log["eval"].append({"step": 0, **evaluate(model, tok, val, ans, yes_id, no_id)})
        print("eval step 0", log["eval"][-1], flush=True)

    losses = []
    while step < total and (time.time() - t0) / 60 < a.minutes:
        idx = order[step * a.batch:(step + 1) * a.batch]
        batch = [train[i] for i in idx]
        for i, it in zip(idx, batch):
            if i not in encs:
                encs[i] = prompts.encode_with_positions(tok, it)["input_ids"]
        T = max(len(encs[i]) for i in idx)
        ids = torch.full((len(idx), T), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, T - len(encs[i]):] = torch.tensor(encs[i]); mask[r, T - len(encs[i]):] = 1
        h = model.base_model.model.model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).last_hidden_state[:, -1]
        logits = model.base_model.model.lm_head(h).float()
        target = torch.tensor([yes_id if it["label"] == 1 else no_id for it in batch], device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss)); step += 1
        if step % 25 == 0:
            log["steps"].append({"step": step, "loss": float(np.mean(losses[-25:])), "minutes": (time.time() - t0) / 60})
            print(log["steps"][-1], flush=True)
        if step % a.ckpt_every == 0:
            model.save_pretrained(out_dir / f"step_{step}")
            log["eval"].append({"step": step, **evaluate(model, tok, val, ans, yes_id, no_id)})
            print("eval", log["eval"][-1], flush=True)
            log_path.write_text(json.dumps(log, indent=1))
        if model.device.type == "mps" and step % 20 == 0:
            torch.mps.empty_cache()
    model.save_pretrained(out_dir / "final")
    log["final_step"] = step
    log["stopped_because"] = "time box" if step < total else "epoch finished"
    log["eval"].append({"step": step, "final": True, **evaluate(model, tok, val, ans, yes_id, no_id)})
    log["total_minutes"] = (time.time() - t0) / 60
    log_path.write_text(json.dumps(log, indent=1))
    print("final", log["eval"][-1], f"steps {step}")


if __name__ == "__main__":
    main()
