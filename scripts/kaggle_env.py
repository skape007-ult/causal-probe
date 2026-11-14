"""Record the software/hardware a remote run used; optionally compare a freshly built cache with
the reference answer-logit differences committed from the laptop cache.

  python scripts/kaggle_env.py                      -> results/kaggle_environment.json
  python scripts/kaggle_env.py --agreement gemma2b  -> results/cache_agreement_gemma2b.json
"""
import argparse
import json
import platform
import subprocess

import numpy as np

from causal_probe import cache
from causal_probe.config import RESULTS, ROOT


def environment(out_name: str = "kaggle_environment.json"):
    import sklearn
    import torch
    import transformers
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "sklearn": sklearn.__version__, "numpy": np.__version__,
            "cuda": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_memory_gb": [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)
                              for i in range(torch.cuda.device_count())]}
    for mod in ("sae_lens", "peft", "bitsandbytes", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:      # optional packages
            info[mod] = f"not available ({type(e).__name__})"
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        info["git_commit"] = None
    (RESULTS / out_name).write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def agreement(model: str):
    """The laptop cache for this model was built on Apple MPS (Gemma: embedding matrix on disk).
    How closely does this machine reproduce its per-item answer logit differences?"""
    ref_path = ROOT / "results" / f"ref_logit_diff_{model}.json"
    if not ref_path.exists():
        print(f"no reference file {ref_path.name}; nothing to compare")
        return
    ref = json.loads(ref_path.read_text())
    out = {"model": model, "reference": ref["source"], "tags": {}}
    for tag, r in ref["tags"].items():
        d = cache.cache_dir(model, tag)
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        here = dict(zip(meta["uids"], meta["logit_diff"]))
        common = [u for u in r["uids"] if u in here]
        a = np.array([here[u] for u in common]); b = np.array([dict(zip(r["uids"], r["logit_diff"]))[u] for u in common])
        out["tags"][tag] = {"n_compared": len(common), "n_reference": len(r["uids"]),
                            "max_abs_diff": float(np.abs(a - b).max()), "mean_abs_diff": float(np.abs(a - b).mean()),
                            "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                            "same_answer_frac": float(((a > 0) == (b > 0)).mean()),
                            "device_here": meta["device"], "dtype_here": meta["model_dtype"]}
        print(tag, out["tags"][tag])
    (RESULTS / f"cache_agreement_{model}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement", default=None)
    ap.add_argument("--out", default="kaggle_environment.json", help="file name under results/ for the environment record")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    agreement(a.agreement) if a.agreement else environment(a.out)
