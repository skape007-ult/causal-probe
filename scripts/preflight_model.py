"""Can this model run on this machine? Asked before any long stage, so that a run that cannot work
ends in minutes with a clear reason instead of using up the session.

  python scripts/preflight_model.py --model gemma9b --check resources    # no download, no model load
  python scripts/preflight_model.py --model gemma9b --check load         # real load + 16 items through the cache code

resources: (1) the GPUs this model needs are present; (2) the token has access to the repository (gated models) and the
           size of the weights is known; (3) the weights fit the per-GPU limits that models.load_model will use;
           (4) the disk has room for the download.
load:      loads the model exactly as every other script does (models.load_model), reports where the weights went and
           how much memory each GPU holds, then runs the first 16 items of the debiased pool through cache.build_cache
           (the code path of the real run, hooks included) into a scratch cache, and checks the result is finite.
           The runner points CAUSAL_PROBE_CACHE at a scratch folder for this check; the scratch cache is deleted first.

Exit code 0: go. Exit code 3: do not run this model here; the reason is the last line printed and the "verdict" in
results/preflight_<model>.json. Any other exit code: the check itself crashed (also treated as "do not run").
The arithmetic for Gemma-2-9B on two T4s is in KAGGLE.md ("Round 3").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

from causal_probe.config import MULTI_GPU, PARAMS_B, QUANT_4BIT, RESULTS, REMOTE_MODELS, MODELS

GIB = 2 ** 30


def fits(weights_gib: float, gpu_total_gib: list[float], caps_gib: list[float], margin_gib: float = 0.5,
         headroom_gib: float = 0.8) -> tuple[bool, str]:
    """Do `weights_gib` of half-precision weights fit GPUs of the given sizes under the per-GPU limits `caps_gib`?
    Each limit must itself fit its GPU with `headroom_gib` to spare (CUDA context, activations), and the limits
    together must exceed the weights by `margin_gib` (accelerate does not fill a device to the last byte)."""
    if len(gpu_total_gib) < len(caps_gib):
        return False, f"needs {len(caps_gib)} GPUs, found {len(gpu_total_gib)}"
    for i, (tot, cap) in enumerate(zip(gpu_total_gib, caps_gib)):
        if cap > tot - headroom_gib:
            return False, f"GPU {i} has {tot:.1f} GiB; the plan puts up to {cap:.0f} GiB of weights on it and needs {headroom_gib:.1f} GiB spare"
    if weights_gib + margin_gib > sum(caps_gib):
        return False, f"weights are {weights_gib:.1f} GiB; the per-GPU limits add up to {sum(caps_gib):.0f} GiB"
    return True, f"weights {weights_gib:.1f} GiB within limits {'+'.join(f'{c:.0f}' for c in caps_gib)} GiB on GPUs of {'+'.join(f'{t:.1f}' for t in gpu_total_gib)} GiB"


def finish(rec: dict, path, ok: bool, verdict: str):
    rec.update(ok=ok, verdict=verdict)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    print(("PREFLIGHT OK: " if ok else "PREFLIGHT SKIP: ") + verdict, flush=True)
    sys.exit(0 if ok else 3)


def resources(name: str, rec: dict, path):
    import torch

    from causal_probe import models
    hf_id = {**MODELS, **REMOTE_MODELS}[name]
    gpus = [torch.cuda.get_device_properties(i).total_memory / GIB for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    rec.update(model=name, hf_id=hf_id, gpus_gib=[round(g, 2) for g in gpus], multi_gpu=name in MULTI_GPU)
    print(f"GPUs: {[round(g, 1) for g in gpus]} GiB", flush=True)
    # (2) access and size, without downloading anything. A refusal (401/403/404, gated) skips the model; a timeout or a
    # server error does not: it is retried, and if the Hub stays unreachable the numbers from the model card are used.
    n_params, refused, last = None, None, None
    for attempt in range(3):
        try:
            from huggingface_hub import HfApi, hf_hub_download
            info = HfApi().model_info(hf_id, files_metadata=True, token=os.environ.get("HF_TOKEN") or None)
            files = [(s.rfilename, s.size or 0) for s in info.siblings]
            wfiles = [f for f in files if f[0].endswith(".safetensors")] or [f for f in files if f[0].endswith(".bin")]
            n_params = getattr(getattr(info, "safetensors", None), "total", None)
            rec.update(weight_files_gib=round(sum(s for _, s in wfiles) / GIB, 2), n_params_from_hub=n_params, gated=str(getattr(info, "gated", None)))
            # the file listing is public even for gated repositories; fetching one small file proves the token has access
            hf_hub_download(hf_id, "config.json", token=os.environ.get("HF_TOKEN") or None)
            last = None
            break
        except Exception as e:
            last = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403, 404) or type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
                refused = e
                break
            print(f"attempt {attempt + 1}: could not ask the Hub ({type(e).__name__}); retrying", flush=True)
            time.sleep(10)
    if refused is not None:
        finish(rec, path, False, f"no access to {hf_id} ({type(refused).__name__}: {str(refused)[:200]}). For a gated model: accept the licence on "
                                 f"https://huggingface.co/{hf_id} with the account the HF_TOKEN belongs to, and attach the secret to this notebook.")
    if last is not None:
        print(f"the Hub could not be asked ({type(last).__name__}); continuing with the parameter count from the model card", flush=True)
        rec.update(hub_unreachable=f"{type(last).__name__}: {str(last)[:160]}")
    params = (n_params or PARAMS_B[name] * 1e9)
    rec.setdefault("weight_files_gib", round(params * 2 / GIB, 2))
    weights_gib = params * (2 if name not in QUANT_4BIT else 0.75) / GIB      # float16; 4-bit NF4 is about 0.7 bytes per parameter
    rec.update(weights_in_memory_gib=round(weights_gib, 2))
    # (3) do the weights fit?
    if name in MULTI_GPU:
        caps = models.multi_gpu_caps_gib(max(len(gpus), 2))
        ok, why = fits(weights_gib, gpus, caps)
        rec.update(per_gpu_limits_gib=caps)
        if not ok:
            finish(rec, path, False, f"{name} does not fit: {why}. It needs two 16 GB GPUs (Kaggle: Accelerator = GPU T4 x2).")
    elif gpus:
        ok, why = fits(weights_gib, gpus[:1], [gpus[0] - 2.0])
        if not ok:
            finish(rec, path, False, f"{name} does not fit one GPU: {why}")
    else:
        why = "no CUDA GPU: the model would run on the CPU or on Apple's GPU (fine for the small models only)"
    # (4) disk for the download
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    os.makedirs(home, exist_ok=True)
    free = shutil.disk_usage(home).free / GIB
    rec.update(disk_free_gib=round(free, 1), hf_home=home)
    need = rec["weight_files_gib"] + 12        # the download, the activation caches of four item sets (about 8 GiB for a 9B model), chunk files
    if free < need:
        finish(rec, path, False, f"the download needs about {need:.0f} GiB in {home}; {free:.0f} GiB are free")
    finish(rec, path, True, f"{why}; {free:.0f} GiB disk free for a {rec['weight_files_gib']:.1f} GiB download")


def load(name: str, rec: dict, path, n_items: int):
    import numpy as np
    import torch

    from causal_probe import cache, models
    t0 = time.time()
    try:
        tok = models.load_tokenizer(name)
        model = models.load_model(name)
    except Exception as e:
        finish(rec, path, False, f"loading {name} failed ({type(e).__name__}: {str(e)[:300]})")
    dmap = getattr(model, "hf_device_map", None)
    place = {}
    for k, v in (dmap or {"(whole model)": str(next(model.parameters()).device)}).items():
        place[str(v)] = place.get(str(v), 0) + 1
    mem = [round(torch.cuda.memory_allocated(i) / GIB, 2) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    rec.update(load_seconds=round(time.time() - t0), modules_per_device=place, gib_allocated_after_load=mem)
    print(f"loaded in {time.time() - t0:.0f} s; modules per device {place}; GiB allocated {mem}", flush=True)
    if name in MULTI_GPU and any(str(v) in ("cpu", "disk") for v in (dmap or {}).values()):
        finish(rec, path, False, f"part of {name} was placed on {sorted({str(v) for v in dmap.values()} & {'cpu', 'disk'})}: it does not fit the GPUs")
    t1 = time.time()
    try:
        # a scratch cache left by an earlier attempt would be found "already complete" and nothing would be run
        shutil.rmtree(cache.cache_dir(name, "c2c_debiased"), ignore_errors=True)
        cache.build_cache(name, "c2c_debiased", model=model, tok=tok, limit=n_items)
        resid, meta, _ = cache.load_cache(name, "c2c_debiased")
        ld = np.array(meta["logit_diff"])
        assert resid.shape[0] == n_items and np.isfinite(np.asarray(resid[:n_items], np.float32)).all() and np.isfinite(ld).all(), "non-finite output"
    except Exception as e:
        finish(rec, path, False, f"{name} loaded but {n_items} items through the cache code failed ({type(e).__name__}: {str(e)[:300]})")
    peak = [round(torch.cuda.max_memory_allocated(i) / GIB, 2) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    per_item = (time.time() - t1) / n_items
    rec.update(gib_peak_during_forward=peak, seconds_per_item_first_batch=round(per_item, 3), resid_shape=list(resid.shape),
               logit_diff_first_items=[round(float(v), 3) for v in ld[:8]],
               projected_minutes_for_9000_items=round(per_item * 9000 / 60, 1))
    finish(rec, path, True, f"{name} loads and runs: peak GiB per GPU {peak}, activations {list(resid.shape)}, "
                            f"about {per_item * 9000 / 60:.0f} min projected for the four item sets (upper bound: includes warm-up)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--check", required=True, choices=["resources", "load"])
    ap.add_argument("--items", type=int, default=16)
    a = ap.parse_args()
    out = RESULTS / f"preflight_{a.model}.json"
    record = json.loads(out.read_text()) if out.exists() and a.check == "load" else {}
    resources(a.model, record, out) if a.check == "resources" else load(a.model, record, out, a.items)
