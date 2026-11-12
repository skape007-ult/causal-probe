"""Environment verification. Writes results/env_check_<model>.json.

1. MPS vs CPU: final-position logits for 5 real prompts must match closely (fp32 CPU is the
   reference; MPS is tried in fp32, fp16 and bf16).
2. Batched (left-padded) vs single-item forward passes must match.
3. Speed benchmark (items/s) per device/dtype.
4. Optional (--tl): TransformerLens vs plain HF logits on the same 5 prompts.
"""
import argparse
import json
import time

import numpy as np
import torch

from causal_probe import data, models, prompts
from causal_probe.config import MODELS, RESULTS


def final_logits(model, tok, texts, device, batch=False):
    outs = []
    if batch:
        enc = tok(texts, add_special_tokens=False, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():   # production path: unembed the last position only
            lg = models.final_logits(model, enc.input_ids, enc.attention_mask).cpu()
        return lg
    for t in texts:
        enc = tok(t, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            outs.append(model(**enc).logits[0, -1].float().cpu())
    return torch.stack(outs)


def compare(a, b, ans):
    da, db = models.answer_logit_diff(a, ans), models.answer_logit_diff(b, ans)
    return {
        "max_abs_logit_diff": float((a - b).abs().max()),
        "max_abs_answer_logit_diff_gap": float((da - db).abs().max()),
        "top1_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
        "any_nan": bool(torch.isnan(b).any()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--tl", action="store_true")
    ap.add_argument("--cpu-dtype", default="float32")
    ap.add_argument("--skip-cpu", action="store_true", help="for models whose fp32 copy does not fit in RAM")
    ap.add_argument("--dtypes", default="float32,float16,bfloat16")
    a = ap.parse_args()

    tok = models.load_tokenizer(a.model)
    ans = models.answer_token_ids(tok)
    items = data.load_c2c("test")[::240][:3] + data.cladder_sample()[::1500][:2]
    texts = [prompts.render(tok, it) for it in items]
    bench_items = data.load_c2c("test")[100:132]
    bench = sorted((prompts.render(tok, it) for it in bench_items), key=len)
    res = {"model": MODELS[a.model], "torch": torch.__version__, "n_prompts": len(texts),
           "answer_tokens": ans, "configs": {}}

    ref = None
    if not a.skip_cpu:
        m = models.load_model(a.model, device="cpu", dtype=getattr(torch, a.cpu_dtype))
        ref = final_logits(m, tok, texts, "cpu")
        res["cpu_batched_vs_single"] = compare(ref, final_logits(m, tok, texts, "cpu", batch=True), ans)
        t0 = time.time()
        final_logits(m, tok, bench[:8], "cpu", batch=True)
        res["cpu_ref_dtype"] = a.cpu_dtype
        res["configs"]["cpu/ref"] = {"items_per_s": 8 / (time.time() - t0)}
        res["ref_answer_logit_diff"] = models.answer_logit_diff(ref, ans).tolist()
        if a.tl:
            try:
                from transformer_lens import TransformerBridge  # TL >= 4: HookedTransformer was removed
                tl = TransformerBridge.boot_transformers(MODELS[a.model], device="cpu", dtype=torch.float32)
                lg = []
                for t in texts:
                    ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids
                    with torch.no_grad():
                        lg.append(tl(ids, return_type="logits")[0, -1].float())
                res["transformer_lens_vs_hf"] = compare(ref, torch.stack(lg), ans)
                del tl
            except Exception as e:  # recorded, not hidden
                res["transformer_lens_vs_hf"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        del m

    if torch.backends.mps.is_available():
        for dn in a.dtypes.split(","):
            key = f"mps/{dn}"
            try:
                m = models.load_model(a.model, device="mps", dtype=getattr(torch, dn))
                lg = final_logits(m, tok, texts, "mps")
                r = compare(ref, lg, ans) if ref is not None else {"any_nan": bool(torch.isnan(lg).any()),
                                                                 "answer_logit_diff": models.answer_logit_diff(lg, ans).tolist()}
                r["batched_vs_single"] = compare(lg, final_logits(m, tok, texts, "mps", batch=True), ans)
                final_logits(m, tok, bench[:8], "mps", batch=True)  # warm-up
                t0 = time.time()
                for i in range(0, 32, 8):
                    final_logits(m, tok, bench[i:i + 8], "mps", batch=True)
                torch.mps.synchronize()
                r["items_per_s"] = 32 / (time.time() - t0)
                r["mps_alloc_gb"] = torch.mps.driver_allocated_memory() / 1e9
                res["configs"][key] = r
                del m
                torch.mps.empty_cache()
            except Exception as e:
                res["configs"][key] = {"error": f"{type(e).__name__}: {e}"[:500]}
    res["mean_bench_tokens"] = float(np.mean([len(tok(t, add_special_tokens=False).input_ids) for t in bench]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"env_check_{a.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
