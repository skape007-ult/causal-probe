"""Write finetune/notebooks/ft_gputest.ipynb: a run of a few minutes that checks the Kaggle path end to end.

It answers five questions and writes the answers to kaggle_out/gputest.json:
  1. which GPUs the session was given (asked for: T4 x2);
  2. whether the internet is on (one small anonymous request to huggingface.co for Qwen2.5-1.5B's config.json);
  3. whether the output directory survives and downloads (results.zip + STATUS.txt, same final check as v1.0);
  4. STATUS.txt is written by the v1.0 FINAL_CHECK cell;
  5. whether the HF_TOKEN secret is readable in a notebook pushed through the API. Only yes/no and the exception
     type are recorded; the value is never printed, stored or sent anywhere.
It also records the versions of torch, transformers and peft in Kaggle's image (the training notebooks pin theirs).

The platform cell and the final cell are the v1.0 ones, imported from scripts/notebook_cells.py unchanged.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from notebook_cells import FINAL_CHECK                         # noqa: E402
from ft_cells import FT_PLATFORM_SETUP                         # noqa: E402

HEAD = '''
import os, subprocess, json, time, zipfile
T0 = time.time()
RUN, COMMIT, NEEDS_TOKEN = "ft_gputest", "%s", False
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout
except Exception:
    gpu = ""
print("first line of this log. GPUs:"); print(gpu or "  none")
'''

BODY = '''
res = {"run": RUN, "commit": COMMIT, "platform": PLATFORM, "gpus": [l.strip() for l in gpu.strip().splitlines()[1:]]}

# internet: one anonymous request (no token is sent)
import urllib.request
try:
    with urllib.request.urlopen("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json", timeout=30) as r:
        res["internet"] = {"ok": True, "hidden_size": json.loads(r.read())["hidden_size"]}
except Exception as e:
    res["internet"] = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:200]}
print("internet:", res["internet"])

# secret: can a notebook pushed through the API read HF_TOKEN? yes/no only
try:
    from kaggle_secrets import UserSecretsClient
    _t = UserSecretsClient().get_secret("HF_TOKEN")
    res["secret_HF_TOKEN"] = {"readable": bool(_t)}
    del _t
except Exception as e:
    res["secret_HF_TOKEN"] = {"readable": False, "error_type": type(e).__name__}
print("HF_TOKEN readable:", res["secret_HF_TOKEN"])

# versions in the image, and one tiny CUDA computation per GPU
import importlib.metadata as md
res["versions"] = {}
for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers", "numpy", "scikit-learn"):
    try:
        res["versions"][p] = md.version(p)
    except Exception:
        res["versions"][p] = None
print("versions:", res["versions"])
try:
    import torch
    res["cuda"] = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(), "checks": []}
    for i in range(torch.cuda.device_count()):
        x = torch.randn(2000, 2000, device=f"cuda:{i}", dtype=torch.float16)
        res["cuda"]["checks"].append({"device": torch.cuda.get_device_name(i), "finite": bool(torch.isfinite((x @ x).float().mean()))})
except Exception as e:
    res["cuda"] = {"error": type(e).__name__ + ": " + str(e)[:200]}
print("cuda:", res["cuda"])

res["disk_free_gb"] = {p: round(__import__("shutil").disk_usage(p).free / 1e9, 1) for p in ("/kaggle/working", "/kaggle/temp", "/tmp") if os.path.isdir(p)}
res["minutes"] = round((time.time() - T0) / 60, 2)
with open(os.path.join(WORK, "gputest.json"), "w") as f:
    json.dump(res, f, indent=1)
with zipfile.ZipFile(os.path.join(WORK, "results.zip"), "w", zipfile.ZIP_STORED) as z:
    z.write(os.path.join(WORK, "gputest.json"), "results/ft_gputest.json")
    z.writestr("results/padding.txt", "x" * 2000)          # the final check wants an archive of at least 1000 bytes
expected, skipped = ["results.zip"], ""
not_finished = "" if (res["gpus"] and res["internet"]["ok"] and res.get("cuda", {}).get("available")) else \\
    "the test did not find what it asked for: gpus=%s internet=%s cuda=%s" % (res["gpus"], res["internet"]["ok"], res.get("cuda"))
'''


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def main():
    commit = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"
    nb = {"cells": [{"cell_type": "markdown", "metadata": {},
                     "source": ["# Fine-tuning study: Kaggle path test (a few minutes, no model is loaded)\n"]},
                    code(HEAD % commit), code(FT_PLATFORM_SETUP), code(BODY), code(FINAL_CHECK)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "finetune" / "notebooks" / "ft_gputest.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
