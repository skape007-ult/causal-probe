"""E7/E8 pilot, GPU side (PREREG Amendment 3, Sections 4 and 5). Reads finetune/e7e8/pilot_inputs.npz (made on the
laptop by e7e8_prepare.py) and applies every vector at (content_end, L_c) of every pilot item; no fitting here.

  python finetune/scripts/e7e8_gpu.py --runs A_s0,B_s0 --adapters <dir> --out <dir>
  ../causal-probe/.venv/bin/python finetune/scripts/e7e8_gpu.py --runs A_s0 --tiny --model qwen0.5b --inputs <synthetic npz> --out <dir>

Section 5: the adapter is merged into a float32 base (merge_and_unload) and the model runs under float16 autocast
(the arithmetic of the E6 and scoring passes). Before anything else, each model must pass
  (a) the logic of v1.0 tests/test_steer.py's three tests (KV-cache = full forward at the final position and at
      content_end; the hook changes exactly the intended layer), tolerance 0.02 in LD;
  (c) for the first 20 pilot items, |(h_stored - h_live).v| <= 0.05 sigma_v for g and every control, and relative norm
      <= 0.5%;
and after the run (b) every zero-vector LD within 0.05 of the stored POOL-LB score. A failed check writes FAILED for
that model and its numbers are not used (the run continues with the next model, so the failure is fully reported).
Outputs in --out/<run>/: ld.npy [items, 97] (column 0 zero vector, 1..92 steering, 93..96 patches), checks.json, DONE
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]
import numpy as np            # noqa: E402
import torch                  # noqa: E402

from causal_probe import models, prompts, steer   # noqa: E402
from causal_probe.cache import ResidCollector      # noqa: E402
from causal_probe.config import MODELS, REMOTE_MODELS   # noqa: E402
from ftstudy import datasets as DS                 # noqa: E402

TOL_A, TOL_B, TOL_C_PROJ, TOL_C_NORM = 0.02, 0.05, 0.05, 0.005


def check_a(model, ids, p_ce, layers, ans, ctx, d):
    """KV-cache = full forward at both positions and three layers; the hook moves exactly the intended layer."""
    g = torch.Generator().manual_seed(0)
    vecs = torch.cat([torch.zeros(1, d), 3.0 * torch.randn(4, d, generator=g)])
    worst = 0.0
    for pos in (None, p_ce):
        for layer in layers:
            with ctx():
                fast = steer.steered_logit_diffs(model, ids, layer, vecs, ans, sub_batch=3, pos=pos)
                slow = steer.steered_logit_diffs_full(model, ids, layer, vecs, ans, pos=pos)
            worst = max(worst, float(np.abs(fast - slow).max()))
    # hooked-layer test (v1.0 test_hook_changes_exactly_the_intended_layer), at the site layer
    L = layers[1]
    T = len(ids)
    t = torch.tensor([ids])
    v = 2.0 * torch.randn(1, d, generator=torch.Generator().manual_seed(1))

    def record(layer=None):
        hook = steer.AddAtLast(model, layer) if layer is not None else None
        if hook:
            hook.vecs = v
        coll = ResidCollector(model)
        coll.idx = torch.tensor([[T - 1, T - 2]])
        with torch.no_grad(), ctx():
            models.body_forward(model, t, torch.ones_like(t))
        if hook:
            hook.remove()
        r = coll.pop()[0]
        coll.remove()
        return r
    r0, r1 = record(), record(L)
    dlt = (r1 - r0)
    hook_ok = bool(torch.allclose(dlt[L, 0], v[0].float(), atol=0.05 * float(v.norm()) / np.sqrt(d) + 1e-2)
                   and float(dlt[:L].abs().max()) < 1e-3 and float(dlt[L, 1].abs().max()) < 1e-3)
    return {"kv_vs_full_max_abs_ld": worst, "kv_ok": worst <= TOL_A, "hook_ok": hook_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--adapters", default="")
    ap.add_argument("--inputs", default=str(ROOT / "finetune" / "e7e8" / "pilot_inputs.npz"))
    ap.add_argument("--prepare", default="")
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true", help="laptop smoke test: 6 items, a random LoRA allowed")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else nullcontext
    hf_id = {**MODELS, **REMOTE_MODELS}[a.model]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    ans = models.answer_token_ids(tok)
    Z = np.load(a.inputs)
    prep = json.loads(Path(a.prepare or str(a.inputs).replace("pilot_inputs", "pilot_prepare").replace(".npz", ".json")).read_text())
    items = DS.materialise("eval_pool_lb")
    idx = Z["pilot_idx"][: 6 if a.tiny else None]
    enc = [prompts.encode_with_positions(tok, items[i]) for i in idx]
    for run in a.runs.split(","):
        out = Path(a.out) / run
        out.mkdir(parents=True, exist_ok=True)
        if (out / "DONE").exists():
            continue
        t0 = time.time()
        L = prep["models"][run]["layer"]
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)
        if run != "base":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(Path(a.adapters) / run)).merge_and_unload()
        model = model.to(device).eval()
        d = model.config.hidden_size
        checks = {"run": run, "layer": L, "device": device}
        n_layers = len(models.get_layers(model))
        checks["a"] = check_a(model, enc[0]["input_ids"], enc[0]["pos"]["content_end"], (0, L, n_layers), ans, ctx, d)
        # (c): stored against live activation at the site, first 20 items
        hc, V, sg = Z[f"{run}|h_check"], Z[f"{run}|check_dirs"].astype(float), Z[f"{run}|check_sigma"].astype(float)
        coll = ResidCollector(model)
        live = []
        for k in range(min(20, len(enc))):
            ids = torch.tensor([enc[k]["input_ids"]])
            coll.idx = torch.tensor([[enc[k]["pos"]["content_end"]]])
            with torch.no_grad(), ctx():
                models.body_forward(model, ids, torch.ones_like(ids))
            live.append(coll.pop()[0, L, 0].numpy())
        coll.remove()
        live = np.array(live, float)
        if a.tiny:
            checks["c"] = {"ok": None, "note": "synthetic inputs: nothing stored to agree with"}
        else:
            dh = hc[: len(live)].astype(float) - live
            proj = np.abs(dh @ V.T) / sg[None, :]
            rel = np.linalg.norm(dh, axis=1) / np.linalg.norm(hc[: len(live)], axis=1)
            checks["c"] = {"max_proj_in_sigma": float(proj.max()), "max_rel_norm": float(rel.max()),
                           "ok": bool(proj.max() <= TOL_C_PROJ and rel.max() <= TOL_C_NORM)}
        ok_pre = checks["a"]["kv_ok"] and checks["a"]["hook_ok"] and checks["c"]["ok"] is not False
        LD = np.full((len(enc), 97), np.nan, np.float32)
        if ok_pre:
            S, P = torch.tensor(Z[f"{run}|steer"].astype(np.float32)), Z[f"{run}|patch"].astype(np.float32)
            if a.tiny:
                S, P = S[:, :d] if S.shape[1] >= d else torch.nn.functional.pad(S, (0, d - S.shape[1])), np.zeros((len(enc), 4, d), np.float32)
            for k, e in enumerate(enc):
                vecs = torch.cat([torch.zeros(1, d), S, torch.tensor(P[k])])
                with torch.no_grad(), ctx():
                    LD[k] = steer.steered_logit_diffs(model, e["input_ids"], L, vecs, ans, sub_batch=16, pos=e["pos"]["content_end"])
                if k % 25 == 0:
                    print(f"[e7e8 {run}] {k}/{len(enc)} items, {time.time() - t0:.0f} s", flush=True)
            stored = Z[f"{run}|stored_ld"][: len(enc)]
            diff = float(np.abs(LD[:, 0] - stored).max())
            checks["b"] = {"max_abs_zero_vs_stored": diff, "ok": bool(diff <= TOL_B) if not a.tiny else None}
        checks["status"] = "OK" if ok_pre and checks.get("b", {}).get("ok") is not False else "FAILED"
        checks["seconds"] = round(time.time() - t0, 1)
        np.save(out / "ld.npy", LD)
        (out / "checks.json").write_text(json.dumps(checks, indent=1))
        (out / "DONE").write_text(checks["status"] + "\n")
        print(f"[e7e8 {run}] {checks['status']}: {json.dumps({k: checks[k] for k in ('a', 'c') })} b={checks.get('b')}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
