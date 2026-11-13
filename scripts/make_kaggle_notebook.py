"""Write notebooks/kaggle_gpu.ipynb, pinned to a commit of this repository.

  python scripts/make_kaggle_notebook.py <commit-sha>

The notebook is deliberately thin: it clones the repo at that commit, installs the pinned
requirements, reads the Hugging Face token from Kaggle Secrets and calls scripts/kaggle_run.py,
which is the same code that was smoke-tested locally.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/skape007-ult/causal-probe.git"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def main(commit: str):
    cells = [
        md("""
# causal-probe: GPU runs (Gemma-2-2B steering / patching / SAEs, and one larger model)

Read `KAGGLE.md` in the repository first. Settings needed on the right-hand panel:
**Accelerator = GPU T4 x2 (or P100)**, **Internet = On**, and a Kaggle Secret named **`HF_TOKEN`**
attached to this notebook (the Gemma weights are gated). Then *Save Version → Save & Run All (Commit)*.

Everything heavy happens in `scripts/kaggle_run.py`. Each stage is time-limited, skipped if already
done, and allowed to fail without stopping the rest. Output: zip files in `/kaggle/working`.
"""),
        code(f"""
# ---- settings ---------------------------------------------------------------
COMMIT       = "{commit}"   # the code this notebook runs (pinned)
RUN_GEMMA    = True          # gemma-2-2b-it: cache, steering, patching, SAE tier
LARGE_MODEL  = "qwen3b"      # "qwen3b" (float16) | "qwen7b-4bit" (4-bit, see KAGGLE.md) | None
SMOKE_TEST   = False         # True: a few dozen items per stage (about 15 minutes) to check the plumbing
EXPORT_RANDOM_INIT = True    # include the random-init control activations in the zips (about +1 GB)
SKIP_STAGES  = ""            # comma-separated stage-name prefixes to skip, e.g. "gemma.gemma2b.sae"
"""),
        code("""
import os, subprocess, sys, json, shutil, time
t_start = time.time()
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else ""
print(gpu or "NO GPU FOUND: set Accelerator to a GPU in the notebook settings (the run would be far too slow on CPU)")

# Hugging Face token from Kaggle Secrets. It is never printed or written to disk by this notebook.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("WARNING: no HF_TOKEN secret available (%s). Gemma stages will fail; the Qwen model needs no token." % type(e).__name__)

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_out")
TEMP = "/kaggle/temp" if os.path.isdir("/kaggle") else os.path.abspath("kaggle_tmp")
os.makedirs(TEMP, exist_ok=True); os.makedirs(WORK, exist_ok=True)
os.environ["HF_HOME"] = TEMP + "/hf"                 # model downloads (not part of the notebook output)
os.environ["CAUSAL_PROBE_CACHE"] = TEMP + "/cache"   # full activation cache (not part of the output)
os.environ["CAUSAL_PROBE_JOBS"] = "4"
os.environ["PYTHONPATH"] = TEMP + "/causal-probe/src"   # the package is used straight from the checkout
"""),
        code(f"""
# ---- code at the pinned commit ------------------------------------------------
REPO_DIR = TEMP + "/causal-probe"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", "{REPO}", REPO_DIR], check=True)
subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=REPO_DIR)
subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO_DIR, check=True)
print("code at", subprocess.run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR, capture_output=True, text=True).stdout)
"""),
        code("""
# ---- pinned environment -------------------------------------------------------
# Level 1: a fresh Python 3.12 virtual environment with the full freeze from the laptop (requirements.txt).
# Level 2: the same venv with the pinned direct dependencies only (requirements-kaggle.txt).
# Level 3: Kaggle's own Python and torch, plus the missing packages (versions then differ; recorded in the results).
def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("FAILED:", cmd, "\\n", (r.stdout + r.stderr)[-1500:])
    return r.returncode == 0

VENV = TEMP + "/venv"
PY, level = None, None
if os.path.exists(VENV + "/bin/python") and os.path.exists(TEMP + "/install_level"):
    PY, level = VENV + "/bin/python", open(TEMP + "/install_level").read()
    if level.startswith("3"): PY = sys.executable
else:
    extra = " bitsandbytes" if LARGE_MODEL and "4bit" in LARGE_MODEL else ""
    if sh(f"{sys.executable} -m pip install -q uv") and sh(f"uv venv --python 3.12 {VENV}"):
        if sh(f"uv pip install -q --python {VENV}/bin/python -r {REPO_DIR}/requirements.txt{extra} && uv pip install -q --python {VENV}/bin/python -e {REPO_DIR}"):
            PY, level = VENV + "/bin/python", "1 (full freeze)"
        elif sh(f"uv pip install -q --python {VENV}/bin/python -r {REPO_DIR}/requirements-kaggle.txt{extra} && uv pip install -q --python {VENV}/bin/python -e {REPO_DIR}"):
            PY, level = VENV + "/bin/python", "2 (pinned direct dependencies)"
    if PY is None:
        sh(f"{sys.executable} -m pip install -q sae-lens peft accelerate{extra}")
        PY, level = sys.executable, "3 (Kaggle's own packages; versions differ from the laptop)"
    open(TEMP + "/install_level", "w").write(level)
print("install level:", level, "| python:", PY, "| %.0f min so far" % ((time.time() - t_start) / 60))
ok = sh(f"{PY} -c \\"import torch, transformers, sklearn, causal_probe; print(torch.__version__, transformers.__version__, torch.cuda.is_available())\\"")
r = subprocess.run([PY, "-c", "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"], capture_output=True, text=True)
print(r.stdout or r.stderr[-800:])
"""),
        code("""
# ---- run (resumable: re-running this cell skips finished stages) ---------------
cmd = [PY, "-u", "scripts/kaggle_run.py", "--out", WORK]
if RUN_GEMMA: cmd.append("--gemma")
if LARGE_MODEL: cmd += ["--large", LARGE_MODEL]
if SMOKE_TEST: cmd.append("--tiny")
if not EXPORT_RANDOM_INIT: cmd.append("--no-random-export")
if SKIP_STAGES: cmd += ["--skip", SKIP_STAGES]
print(" ".join(cmd))
p = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="")
p.wait()
print("runner exit code", p.returncode, "| total %.0f min" % ((time.time() - t_start) / 60))
"""),
        code("""
# ---- what to download ---------------------------------------------------------
st = REPO_DIR + "/results/kaggle_run_status.json"
if os.path.exists(st):
    s = json.load(open(st))
    for k, v in s["stages"].items():
        print(("ok     " if v["ok"] else "FAILED ") + f"{v['minutes']:6.1f} min  {k}" + ("" if v["ok"] else "   <- " + v["detail"]))
    print("export:", json.dumps(s.get("export", {}).get("zip_megabytes", {})))
for f in sorted(os.listdir(WORK)):
    if f.endswith(".zip"):
        print(f"{WORK}/{f}  {os.path.getsize(WORK + '/' + f) / 1e6:.1f} MB")
"""),
    ]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = ROOT / "notebooks" / "kaggle_gpu.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, "pinned to", commit)


if __name__ == "__main__":
    main(sys.argv[1])
