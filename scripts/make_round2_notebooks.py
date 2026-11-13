"""Write the round-2 GPU notebooks, pinned to a commit of this repository.

  python scripts/make_round2_notebooks.py <commit-sha>

One notebook per run, each a single "Run all". All of them are thin wrappers around
scripts/gpu_round2.py, which is the code that was smoke-tested locally. See KAGGLE.md, "Round 2".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/skape007-ult/causal-probe.git"

RUNS = [  # file name, platform, run name, models, jobs, fallback, needs HF token, title, what it does, estimate
    dict(file="kaggle_r2_qwen3b.ipynb", platform="kaggle", run="r2_qwen3b", models="qwen3b",
         jobs="cache,controls,controls_hyp,feature,patch_same,patch,export", fallback="", token=False,
         title="Round 2 / A: Qwen2.5-3B interventions",
         what="matched steering controls at the final and end-of-hypothesis tokens, the adjacency causal test at four sites, "
              "label-preserving and label-changing patching pairs", est="about 3 hours"),
    dict(file="kaggle_r2_qwen7b.ipynb", platform="kaggle", run="r2_qwen7b", models="qwen7b-4bit",
         jobs="behaviour,export", fallback="qwen7b-4bit:qwen7b", token=False,
         title="Round 2 / B: Qwen2.5-7B behaviour check",
         what="per-item answer logits of the 7B model (4-bit; float16 over two GPUs if 4-bit loading fails) on the debiased pool, "
              "the official Corr2Cause test set, CLadder and the original probe pool; no interventions", est="1.5 to 3 hours"),
    dict(file="kaggle_r2_gemma.ipynb", platform="kaggle", run="r2_gemma", models="gemma2b",
         jobs="cache,controls,controls_hyp,feature,patch_same,export", fallback="", token=True,
         title="Round 2 / D: Gemma-2-2B interventions",
         what="matched steering controls at both tokens, the adjacency causal test, label-preserving patching pairs", est="about 2.5 hours"),
    dict(file="colab_r2_qwen1.5b.ipynb", platform="colab", run="r2_qwen15b", models="qwen1.5b",
         jobs="cache,controls_hyp,feature,patch_same,export", fallback="", token=False,
         title="Round 2 / C: Qwen2.5-1.5B interventions (Colab)",
         what="steering controls at the end-of-hypothesis token, the adjacency causal test, label-preserving patching pairs "
              "(this model's final-token controls already ran on the laptop)", est="about 1.5 to 2 hours"),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


KAGGLE_SETUP = '''
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
    print("No HF_TOKEN secret available (%s)." % type(e).__name__, "This run needs it (Gemma is gated)." if NEEDS_TOKEN else "This run does not need one.")

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_out")
TEMP = "/kaggle/temp" if os.path.isdir("/kaggle") else os.path.abspath("kaggle_tmp")
'''

COLAB_SETUP = '''
import os, subprocess, sys, json, shutil, time
t_start = time.time()
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else ""
print(gpu or "NO GPU FOUND: Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all")

# Hugging Face token from Colab Secrets (the key icon in the left bar), only if this run needs one.
# It is never printed or written to disk by this notebook.
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded from Colab Secrets")
except Exception as e:
    print("No HF_TOKEN secret available (%s)." % type(e).__name__, "This run needs it (Gemma is gated)." if NEEDS_TOKEN else "This run does not need one.")

WORK = "/content/out" if os.path.isdir("/content") else os.path.abspath("kaggle_out")
TEMP = "/content/tmp" if os.path.isdir("/content") else os.path.abspath("kaggle_tmp")
'''

COMMON_SETUP = '''
os.makedirs(TEMP, exist_ok=True); os.makedirs(WORK, exist_ok=True)
os.environ["HF_HOME"] = TEMP + "/hf"                 # model downloads (not part of the output)
os.environ["CAUSAL_PROBE_CACHE"] = TEMP + "/cache"   # full activation cache (not part of the output)
os.environ["CAUSAL_PROBE_JOBS"] = "4" if (os.cpu_count() or 2) >= 4 else "2"
os.environ["PYTHONPATH"] = TEMP + "/causal-probe/src"   # the package is used straight from the checkout
'''

CLONE = f'''
# ---- code at the pinned commit ------------------------------------------------
REPO_DIR = TEMP + "/causal-probe"
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--quiet", "{REPO}", REPO_DIR], check=True)
subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=REPO_DIR)
subprocess.run(["git", "checkout", "--quiet", COMMIT], cwd=REPO_DIR, check=True)
print("code at", subprocess.run(["git", "log", "--oneline", "-1"], cwd=REPO_DIR, capture_output=True, text=True).stdout)
'''

ENV = '''
# ---- pinned environment -------------------------------------------------------
# Level 1: a fresh Python 3.12 virtual environment with the full freeze from the laptop (requirements.txt).
# Level 2: the same venv with the pinned direct dependencies only (requirements-kaggle.txt).
# Level 3: this machine's own Python and torch, plus the missing packages (versions then differ; recorded in the results).
def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("FAILED:", cmd, "\\n", (r.stdout + r.stderr)[-1500:])
    return r.returncode == 0

VENV = TEMP + "/venv"
PY, level = None, None
extra = " bitsandbytes" if "4bit" in MODELS else ""
if os.path.exists(VENV + "/bin/python") and os.path.exists(TEMP + "/install_level"):
    PY, level = VENV + "/bin/python", open(TEMP + "/install_level").read()
    if level.startswith("3"): PY = sys.executable
else:
    if sh(f"{sys.executable} -m pip install -q uv") and sh(f"uv venv --python 3.12 {VENV}"):
        if sh(f"uv pip install -q --python {VENV}/bin/python -r {REPO_DIR}/requirements.txt{extra} && uv pip install -q --python {VENV}/bin/python -e {REPO_DIR}"):
            PY, level = VENV + "/bin/python", "1 (full freeze)"
        elif sh(f"uv pip install -q --python {VENV}/bin/python -r {REPO_DIR}/requirements-kaggle.txt{extra} && uv pip install -q --python {VENV}/bin/python -e {REPO_DIR}"):
            PY, level = VENV + "/bin/python", "2 (pinned direct dependencies)"
    if PY is None:
        sh(f"{sys.executable} -m pip install -q sae-lens peft accelerate{extra}")
        PY, level = sys.executable, "3 (this machine's own packages; versions differ from the laptop)"
    open(TEMP + "/install_level", "w").write(level)
print("install level:", level, "| python:", PY, "| %.0f min so far" % ((time.time() - t_start) / 60))
r = subprocess.run([PY, "-c", "import torch, transformers, sklearn, causal_probe; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available(), '| GPUs', torch.cuda.device_count())"], capture_output=True, text=True)
print(r.stdout or r.stderr[-800:])
if extra:
    r = subprocess.run([PY, "-c", "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)"], capture_output=True, text=True)
    print(r.stdout or "bitsandbytes not importable (the float16 two-GPU fallback will be used): " + r.stderr[-300:])
'''

RUN = '''
# ---- run (resumable: re-running this cell skips finished stages) ---------------
cmd = [PY, "-u", "scripts/gpu_round2.py", "--run", RUN_NAME, "--models", MODELS, "--jobs", JOBS, "--out", WORK]
if FALLBACK: cmd += ["--fallback", FALLBACK]
if SMOKE_TEST: cmd.append("--tiny")
if SKIP_STAGES: cmd += ["--skip", SKIP_STAGES]
print(" ".join(cmd))
p = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="")
p.wait()
print("runner exit code", p.returncode, "| total %.0f min" % ((time.time() - t_start) / 60))
'''

SUMMARY = '''
# ---- what to download ---------------------------------------------------------
st = REPO_DIR + "/results/round2_status_" + RUN_NAME + ".json"
if os.path.exists(st):
    s = json.load(open(st))
    for k in s.get("planned", []):
        v = s["stages"].get(k)
        print("not run         " + k if v is None else ("ok     " if v["ok"] else "FAILED ") + f"{v['minutes']:6.1f} min  {k}" + ("" if v["ok"] else "   <- " + v["detail"]))
    print("export:", json.dumps(s.get("export", {}).get("zip_megabytes", {})))
zips = sorted(f for f in os.listdir(WORK) if f.endswith(".zip"))
for f in zips:
    print(f"{WORK}/{f}  {os.path.getsize(WORK + '/' + f) / 1e6:.1f} MB")
'''

COLAB_DOWNLOAD = '''
# ---- Colab only: hand the zip files to the browser ----------------------------
# Colab deletes /content when the session ends, so download now. If the browser blocks the
# downloads, open the Files panel (folder icon, left), go to /content/out and download from there.
try:
    from google.colab import files
    for f in zips:
        files.download(WORK + "/" + f)
except Exception as e:
    print("automatic download not available here:", type(e).__name__)
'''


def notebook(r: dict, commit: str) -> dict:
    kag = r["platform"] == "kaggle"
    head = f"""
# causal-probe: {r['title']}

Read `KAGGLE.md` ("Round 2") in the repository first. **What this run does:** {r['what']}. **Estimated time:** {r['est']} on a T4.

""" + ("""Settings (right-hand panel): **Accelerator = GPU T4 x2**, **Internet = On**""" +
       (""", and the Kaggle Secret **`HF_TOKEN`** attached to this notebook (Gemma is gated)""" if r["token"] else """ (no Hugging Face token needed)""") +
       """. Then *Save Version → Save & Run All (Commit)*. Output: zip files in `/kaggle/working`.""" if kag else
       """Runtime → Change runtime type → **T4 GPU**; then Runtime → **Run all**. No Hugging Face token needed. Keep this browser tab open until
the last cell has finished: a free Colab session stops when the tab is closed or idle for long, and its files are deleted. The last cell
downloads the zip files through the browser (allow multiple downloads when asked).""") + """

Everything heavy happens in `scripts/gpu_round2.py`: each stage is time-limited, skipped if already done, and allowed to fail without
stopping the rest. Afterwards put the downloaded files into `kaggle_in/""" + r["run"] + """/` on the laptop (see `notes/kaggle_ingest.md`).
"""
    cells = [
        md(head),
        code(f"""
# ---- settings ---------------------------------------------------------------
COMMIT      = "{commit}"   # the code this notebook runs (pinned)
RUN_NAME    = "{r['run']}"
MODELS      = "{r['models']}"
JOBS        = "{r['jobs']}"
FALLBACK    = "{r['fallback']}"
NEEDS_TOKEN = {r['token']}
SMOKE_TEST  = False        # True: a few items per stage (about 10 minutes) to check the plumbing
SKIP_STAGES = ""           # comma-separated stage-name prefixes to skip, e.g. "r2.qwen3b.patch"
"""),
        code((KAGGLE_SETUP if kag else COLAB_SETUP) + COMMON_SETUP),
        code(CLONE), code(ENV), code(RUN), code(SUMMARY),
    ]
    if not kag:
        cells.append(code(COLAB_DOWNLOAD))
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, **({} if kag else {"accelerator": "GPU", "colab": {"gpuType": "T4"}})}}


def main(commit: str):
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for r in RUNS:
        (out / r["file"]).write_text(json.dumps(notebook(r, commit), indent=1))
        print("wrote", out / r["file"], "pinned to", commit[:12])


if __name__ == "__main__":
    main(sys.argv[1])
