"""Write the round-2 GPU notebooks, pinned to a commit of this repository.

  python scripts/make_round2_notebooks.py <commit-sha>

One notebook per run, each a single "Run all". All of them are thin wrappers around
scripts/gpu_round2.py, which is the code that was smoke-tested locally. See KAGGLE.md, "Round 2".
"""
import json
import sys
from pathlib import Path

from notebook_cells import FINAL_CHECK, PLATFORM_SETUP

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/skape007-ult/causal-probe.git"

RUNS = [  # file name, platform, run name, models, jobs, fallback, needs HF token, title, what it does, estimate
    dict(file="kaggle_r2_qwen3b.ipynb", platform="kaggle", run="r2_qwen3b", models="qwen3b",
         jobs="cache,controls,controls_hyp,adjacency,patch_same,patch,export", fallback="", token=False,
         title="Round 2 / A: Qwen2.5-3B interventions",
         what="matched steering controls at the final and end-of-hypothesis tokens, the adjacency causal test at four sites, "
              "label-preserving and label-changing patching pairs", est="3 to 4.5 hours"),
    dict(file="kaggle_r2_qwen7b.ipynb", platform="kaggle", run="r2_qwen7b", models="qwen7b-4bit",
         jobs="behaviour,export", fallback="qwen7b-4bit:qwen7b", token=False,
         title="Round 2 / B: Qwen2.5-7B behaviour check",
         what="per-item answer logits of the 7B model (4-bit; float16 over two GPUs if 4-bit loading fails) on the debiased pool, "
              "the official Corr2Cause test set, CLadder and the original probe pool; no interventions", est="1.5 to 3 hours"),
    dict(file="kaggle_r2_gemma.ipynb", platform="kaggle", run="r2_gemma", models="gemma2b",
         jobs="cache,controls,controls_hyp,adjacency,patch_same,export", fallback="", token=True,
         gated="google/gemma-2-2b-it",
         title="Round 2 / D: Gemma-2-2B interventions",
         what="matched steering controls at both tokens, the adjacency causal test, label-preserving patching pairs", est="2.5 to 4 hours"),
    dict(file="kaggle_r2b_qwen3b_adjacency.ipynb", platform="kaggle", run="r2b_qwen3b", models="qwen3b",
         jobs="cache,controls,controls_hyp,adjacency,export", fallback="", token=False,
         title="Round 2 / A2: Qwen2.5-3B follow-up (corrected steering controls and adjacency test)",
         what="ONLY needed if notebook A was launched from a commit before cdb08d7: steering controls shuffled within hypothesis string, and the adjacency test with all its arms, at four sites",
         est="1.5 to 2.5 hours"),
    dict(file="kaggle_r2b_gemma_adjacency.ipynb", platform="kaggle", run="r2b_gemma", models="gemma2b",
         jobs="cache,controls,controls_hyp,adjacency,export", fallback="", token=True,
         gated="google/gemma-2-2b-it",
         title="Round 2 / D2: Gemma-2-2B follow-up (corrected steering controls and adjacency test)",
         what="ONLY needed if notebook D was launched from a commit before cdb08d7: steering controls shuffled within hypothesis string, and the adjacency test with all its arms, at four sites",
         est="1.5 to 2.5 hours"),
    dict(file="colab_r2b_qwen1.5b_adjacency.ipynb", platform="colab", run="r2b_qwen15b", models="qwen1.5b",
         jobs="cache,controls,controls_hyp,adjacency,export", fallback="", token=False,
         title="Round 2 / C2: Qwen2.5-1.5B follow-up (corrected steering controls and adjacency test; Colab)",
         what="NOT needed if notebook C is run from this commit or from cdb08d7 (it already contains all of this). Only for a C "
              "run from an earlier pin: steering controls shuffled within hypothesis string, and the adjacency test with all its arms, at four sites",
         est="about 1 hour"),
    # C exists twice, for the two platforms; both are the same run (r2_qwen15b) and either file works on either platform.
    # One run of C replaces the original C and its follow-up C2: the code since bab7d1b has the fifth audit's corrections.
    dict(file="kaggle_r2_qwen1.5b.ipynb", platform="kaggle", run="r2_qwen15b", models="qwen1.5b",
         jobs="cache,controls,controls_hyp,adjacency,patch_same,export", fallback="", token=False,
         title="Round 2 / C: Qwen2.5-1.5B interventions",
         what="matched steering controls at both tokens (the debiased-pool controls shuffled within hypothesis string), the adjacency "
              "test with all its arms at four sites, label-preserving patching pairs. One run; no follow-up needed",
         est="about 1.25 hours (the same twelve stages took 72 minutes on Kaggle on 13 Nov)"),
    dict(file="colab_r2_qwen1.5b.ipynb", platform="colab", run="r2_qwen15b", models="qwen1.5b",
         jobs="cache,controls,controls_hyp,adjacency,patch_same,export", fallback="", token=False,
         title="Round 2 / C: Qwen2.5-1.5B interventions (Colab)",
         what="matched steering controls at both tokens (the debiased-pool controls shuffled within hypothesis string), the adjacency "
              "test with all its arms at four sites, label-preserving patching pairs. One run; no follow-up needed",
         est="1.25 to 2 hours"),
    # ---------------- round 3 (Fri 13 Nov, late): depth on the three headline claims; see KAGGLE.md, "Round 3"
    dict(file="kaggle_r3_gemma9b.ipynb", platform="kaggle", run="r3_gemma9b", models="gemma9b",
         jobs="preflight,behaviour,export", fallback="", token=True, gated="google/gemma-2-9b-it", min_gpus=2,
         title="Round 3 / E: Gemma-2-9B behaviour and debiased-pool activations",
         what="a second model family above 7B: per-item answer logits of Gemma-2-9B (float16, split over the two T4s) on the debiased "
              "pool, the official Corr2Cause test set, CLadder and the original pool, and its debiased-pool activations for the "
              "probe comparison on the laptop; no interventions. The first cell checks the token's access to the gated weights "
              "and that two GPUs are present; the runner then checks memory and disk, loads the model and runs 16 items. If any "
              "of that fails the run ends within minutes with the reason (STATUS.txt: SKIPPED)", est="2 to 3.5 hours"),
    dict(file="kaggle_r3_patch240_qwen3b.ipynb", platform="kaggle", run="r3p_qwen3b", models="qwen3b",
         jobs="patch_same,patch,export", fallback="", token=False, extra="--n-pairs 240 --no-resid",
         title="Round 3 / F: Qwen2.5-3B patching on all available pairs",
         what="activation patching on up to 240 label-preserving and 240 label-changing pairs per kind (the pool holds about 230 "
              "and 210) instead of 60, to narrow the interval on their comparison. No component patching, no random-donor control",
         est="1 to 2.5 hours (to be narrowed with the stage times of run A)"),
    dict(file="kaggle_r3_patch240_gemma.ipynb", platform="kaggle", run="r3p_gemma", models="gemma2b",
         jobs="patch_same,patch,export", fallback="", token=True, gated="google/gemma-2-2b-it", extra="--n-pairs 240 --no-resid",
         title="Round 3 / F: Gemma-2-2B patching on all available pairs",
         what="activation patching on up to 240 label-preserving and 240 label-changing pairs per kind (the pool holds about 230 "
              "and 210) instead of 60, to narrow the interval on their comparison. No component patching, no random-donor control",
         est="1 to 2.5 hours (to be narrowed with the stage times of run D)"),
    dict(file="kaggle_r3_controls100_qwen0.5b.ipynb", platform="kaggle", run="r3c_qwen05b", models="qwen0.5b",
         jobs="cache,controls,controls_hyp,adjacency,export", fallback="", token=False, 
         extra="--n-shuf 100 --sites best --control-tags c2c_debiased --no-resid",
         title="Round 3 / G: Qwen2.5-0.5B with 100 control directions",
         what="ONLY if the 20-control run of this model shows an effect beyond its controls (I will say so after its ingest). The steering "
              "controls on the debiased pool and the adjacency test at the two best-layer sites, with 100 shuffled-label control "
              "directions instead of 20, so that a result near the edge of 20 controls can be placed among 100. Same sites as the "
              "20-control run", est="about 1 hour"),
    dict(file="kaggle_r3_controls100_qwen1.5b.ipynb", platform="kaggle", run="r3c_qwen15b", models="qwen1.5b",
         jobs="cache,controls,controls_hyp,adjacency,export", fallback="", token=False, 
         extra="--n-shuf 100 --sites best --control-tags c2c_debiased --no-resid",
         title="Round 3 / G: Qwen2.5-1.5B with 100 control directions",
         what="ONLY if the 20-control run of this model shows an effect beyond its controls (I will say so after its ingest). The steering "
              "controls on the debiased pool and the adjacency test at the two best-layer sites, with 100 shuffled-label control "
              "directions instead of 20, so that a result near the edge of 20 controls can be placed among 100. Same sites as the "
              "20-control run", est="1.5 to 2 hours"),
    dict(file="kaggle_r3_controls100_qwen3b.ipynb", platform="kaggle", run="r3c_qwen3b", models="qwen3b",
         jobs="cache,controls,controls_hyp,adjacency,export", fallback="", token=False, 
         extra="--n-shuf 100 --sites best --control-tags c2c_debiased --no-resid",
         title="Round 3 / G: Qwen2.5-3B with 100 control directions",
         what="ONLY if the 20-control run of this model shows an effect beyond its controls (I will say so after its ingest). The steering "
              "controls on the debiased pool and the adjacency test at the two best-layer sites, with 100 shuffled-label control "
              "directions instead of 20, so that a result near the edge of 20 controls can be placed among 100. Same sites as the "
              "20-control run", est="2.5 to 4.5 hours"),
    dict(file="kaggle_r3_controls100_gemma.ipynb", platform="kaggle", run="r3c_gemma", models="gemma2b",
         jobs="cache,controls,controls_hyp,adjacency,export", fallback="", token=True, gated="google/gemma-2-2b-it",
         extra="--n-shuf 100 --sites best --control-tags c2c_debiased --no-resid",
         title="Round 3 / G: Gemma-2-2B with 100 control directions",
         what="ONLY if the 20-control run of this model shows an effect beyond its controls (I will say so after its ingest). The steering "
              "controls on the debiased pool and the adjacency test at the two best-layer sites, with 100 shuffled-label control "
              "directions instead of 20, so that a result near the edge of 20 controls can be placed among 100. Same sites as the "
              "20-control run", est="2.5 to 4.5 hours"),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SETUP_HEAD = '''
import os, subprocess, sys, json, shutil, time
t_start = time.time()
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else ""
print(gpu or "NO GPU FOUND: switch the accelerator to a GPU (Kaggle: notebook settings; Colab: Runtime -> Change runtime type). The run would be far too slow on CPU.")
'''

COMMON_SETUP = '''
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
extra = ""
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
if "4bit" in MODELS:
    # 4-bit loading needs bitsandbytes. Installed separately and allowed to fail, so that it cannot
    # change the pinned packages above; without it the runner falls back to float16 on two GPUs.
    if PY != sys.executable:
        sh(f"uv pip install -q --no-deps --python {PY} bitsandbytes")
    else:
        sh(f"{sys.executable} -m pip install -q --no-deps bitsandbytes")
    r = subprocess.run([PY, "-c", "import torch, bitsandbytes; print('bitsandbytes', bitsandbytes.__version__, '| torch still', torch.__version__)"], capture_output=True, text=True)
    print(r.stdout or "bitsandbytes not importable (the float16 two-GPU fallback will be used): " + r.stderr[-300:])
'''

RUN = '''
# ---- run (resumable: re-running this cell skips finished stages) ---------------
cmd = [PY, "-u", "scripts/gpu_round2.py", "--run", RUN_NAME, "--models", MODELS, "--jobs", JOBS, "--out", WORK]
if FALLBACK: cmd += ["--fallback", FALLBACK]
cmd += EXTRA_ARGS.split()
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
# ---- what ran, and which archives to expect -----------------------------------
# Nothing in this cell may stop the notebook: the last cell must run whatever happened before.
RUN = RUN_NAME + ("_tiny" if SMOKE_TEST else "")      # a smoke test runs under its own name and results folder
s, expected, zips, not_finished, skipped = {}, ["results_" + RUN + ".zip"], [], "", ""
try:
    st = REPO_DIR + "/results/" + ("_tiny/" if SMOKE_TEST else "") + "round2_status_" + RUN + ".json"
    s = json.load(open(st)) if os.path.exists(st) else {}
    for k in s.get("planned", []):
        v = s["stages"].get(k)
        print("not run         " + k if v is None else ("ok     " if v["ok"] else "FAILED ") + f"{v['minutes']:6.1f} min  {k}" + ("" if v["ok"] else "   <- " + v["detail"]))
    print("export:", json.dumps(s.get("export", {}).get("zip_megabytes", {})))
    if not s.get("finished"):
        not_finished = "the runner did not reach its export step (no finished flag in " + st + ")"
    # expected: the small results archive, every archive the runner recorded, and one activation archive per model whose cache stage succeeded
    expected += list(s.get("export", {}).get("zip_megabytes", {}))
    for m in s.get("models_run", []):
        if any(s.get("stages", {}).get(f"r2.{m}.{j}", {}).get("ok") for j in ("cache", "behaviour.cache")):
            expected.append(f"cache_{RUN}_{m}.zip")
    expected = list(dict.fromkeys(expected))
    zips = sorted(f for f in os.listdir(WORK) if f.endswith(".zip"))
    # a run that decided not to run (no access, not enough GPU memory): see the runner's preflight
    skipped = "; ".join(f"{m}: {why}" for m, why in s.get("skipped", {}).items()) if not s.get("models_run") else ""
except Exception as e:
    not_finished = "the runner's status could not be read (%s: %s)" % (type(e).__name__, e)
# the runner itself ended with an error (an unknown option, a refused output directory): whatever status is on disk is from an earlier attempt
_rc = getattr(globals().get("p"), "returncode", None)
if _rc not in (0, None):
    not_finished = "the runner ended with exit code %s (see the output of the cell above)" % _rc
elif _rc is None:
    not_finished = not_finished or "the run cell was not executed in this session"
'''


def notebook(r: dict, commit: str) -> dict:
    kag = r["platform"] == "kaggle"
    token = ("the secret **`HF_TOKEN`** attached to THIS notebook (Gemma is gated; secrets are attached per notebook, and the "
             "first code cell stops at once if it is missing or has no access to the weights)" if r["token"] else
             "no Hugging Face token needed (if your `HF_TOKEN` secret is attached to this notebook it is used; otherwise the "
             "public Qwen weights are downloaded anonymously and the first cell says so)")
    kaggle_how = ("**On Kaggle:** settings (right-hand panel) **Accelerator = GPU T4 x2**, **Internet = On**, " + token +
                  ". Then *Save Version → Save & Run All (Commit)*. Output: the zip files in **`/kaggle/working/kaggle_out`** "
                  "(Output tab of the saved version, folder `kaggle_out`).")
    colab_how = ("**On Colab:** Runtime → Change runtime type → **T4 GPU**; then Runtime → **Run all**; " + token + ". Keep the "
                 "browser tab open until the last cell has finished: a free Colab session stops when the tab is closed or idle "
                 "for long, and its files are deleted. A cell near the end downloads the zip files from `/content/out` through "
                 "the browser (allow multiple downloads when asked).")
    head = f"""
# causal-probe: {r['title']}

Read `KAGGLE.md` ("Round 2") in the repository first. **What this run does:** {r['what']}. **Estimated time:** {r['est']} on a T4.

""" + (kaggle_how + "\n\n" + colab_how if kag else colab_how + "\n\n" + kaggle_how) + """

The notebook works on either platform: the first code cell detects where it is running, writes the archives to the place that
platform keeps, and stops before any GPU time is spent if it cannot. **The last cell lists the output directory with sizes,
writes `STATUS.txt` there (first line `OK`, `FAILED` or `SKIPPED`) and prints a banner. Download `STATUS.txt` with the zip files.**

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
EXTRA_ARGS  = "{r.get('extra', '')}"   # further options of scripts/gpu_round2.py (size of the intervention jobs)
NEEDS_TOKEN = {r['token']}
GATED_REPOS = "{r.get('gated', '')}"   # the first cell checks that the token has access to these before anything is installed
MIN_GPUS    = {r.get('min_gpus', 1)}
SMOKE_TEST  = False        # True: a few items per stage (about 10 minutes) to check the plumbing
SKIP_STAGES = ""           # comma-separated stage-name prefixes to skip, e.g. "r2.qwen3b.patch"
"""),
        code(SETUP_HEAD + PLATFORM_SETUP + COMMON_SETUP),
        code(CLONE), code(ENV), code(RUN), code(SUMMARY),
        code(FINAL_CHECK),          # always the last cell: lists the output directory, writes STATUS.txt, never raises
    ]
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
