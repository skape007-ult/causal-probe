"""Write the E6 notebook (residual stream of base, A_s0-4, B_s0-4 on POOL-LB), pinned to a pushed commit of exp/finetune.

  ../causal-probe/.venv/bin/python finetune/scripts/make_e6_notebook.py <commit-sha>

Same cells as make_ft_notebooks.py (platform, clone at the pinned commit, v1.0 pinned environment, data at pinned
revisions with every item set hash-checked, v1.0 final check). The adapters are not uploaded again: the notebook is
pushed with the five A/B pair kernels as kernel sources (queue entry `kernel_sources`), finds their adapter archives
under /kaggle/input, and checks every adapter file against finetune/e6/adapter_sha256.json (hashes of the ingested
adapters in finetune/ft_adapters/) before anything runs. One archive per model, written and its work folder deleted as
soon as that model is done, so the output never holds a model twice (about 0.84 GB per model, 9.2 GB in all).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "finetune" / "scripts")]
from notebook_cells import FINAL_CHECK                                              # noqa: E402
from ft_cells import FT_PLATFORM_SETUP                                             # noqa: E402
from make_round2_notebooks import CLONE, COMMON_SETUP, ENV, SETUP_HEAD, code, md     # noqa: E402
from make_ft_notebooks import DATA                                                 # noqa: E402

# PREREG order (base, A, B), interleaved by seed so that a stop leaves A and B with equal seed counts
RUNS = ["base"] + [f"{c}_s{s}" for s in range(5) for c in ("A", "B")]
KERNEL_SOURCES = [f"skape007/ft-pair-a{s}-b{s}" for s in range(5)]

ADAPTERS = '''
# ---- adapters: from the attached outputs of the pair kernels, every file hash-checked ----
import glob, hashlib, zipfile
AD = TEMP + "/adapters"
want = json.load(open(REPO_DIR + "/finetune/e6/adapter_sha256.json"))
zips = sorted(glob.glob("/kaggle/input/**/adapters_ft_pair_A*_B*.zip", recursive=True))
print("adapter archives found:", zips)
for z in zips:
    with zipfile.ZipFile(z) as f:
        f.extractall(TEMP)                       # -> TEMP/ft_adapters/<run>/...
for d in glob.glob("/kaggle/input/**/ft_adapters/*_s[0-9]", recursive=True):    # in case Kaggle unpacked the archives
    if not os.path.isdir(TEMP + "/ft_adapters/" + os.path.basename(d)):
        shutil.copytree(d, TEMP + "/ft_adapters/" + os.path.basename(d))
os.makedirs(AD, exist_ok=True)
for run in [r for r in RUNS if r != "base"]:
    src = TEMP + "/ft_adapters/" + run
    assert os.path.isdir(src), "adapter missing for " + run + ": nothing is run"
    for name, h in want[run].items():
        got = hashlib.sha256(open(src + "/" + name, "rb").read()).hexdigest()
        assert got == h, f"{run}/{name}: sha256 {got[:16]} is not the ingested adapter's {h[:16]}; nothing is run"
    shutil.copytree(src, AD + "/" + run, dirs_exist_ok=True)
print("every adapter matches the ingested one |", "%.0f min so far" % ((time.time() - t_start) / 60))
'''

RUN = '''
# ---- run: one process per GPU; each model is archived as soon as it is done (re-running this cell resumes) ----
import threading, zipfile
OUT = WORK + "/work"
os.makedirs(OUT, exist_ok=True)
n_gpu = max(1, len([l for l in gpu.strip().splitlines()[1:] if l.strip()]))
failed, expected = [], []
def archive(run):
    zp = WORK + "/e6_" + run + ".zip"
    with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for f in ("resid.npy", "meta.json"):
            z.write(OUT + "/" + run + "/" + f, "e6/" + run + "/" + f)
    os.replace(zp + ".tmp", zp)
    shutil.rmtree(OUT + "/" + run, ignore_errors=True)
def worker(g, runs):
    for run in runs:
        if os.path.exists(WORK + "/e6_" + run + ".zip"):
            continue
        cmd = [PY, "-u", "finetune/scripts/e6_extract.py", "--runs", run, "--adapters", AD, "--out", OUT]
        p = subprocess.Popen(cmd, cwd=REPO_DIR, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in p.stdout:
            tail = (tail + [line])[-25:]
            if line.startswith("["): print("gpu%d %s" % (g, line), end="")
        if p.wait() != 0:
            failed.append("%s: exit code %s" % (run, p.returncode))
            print("gpu%d FAILED %s\\n%s" % (g, run, "".join(tail)))
            continue
        archive(run)
threads = [threading.Thread(target=worker, args=(g, RUNS[g::n_gpu])) for g in range(n_gpu)]
[t.start() for t in threads]; [t.join() for t in threads]
print("failed:", failed or "none", "| total %.0f min" % ((time.time() - t_start) / 60))
'''

PACK = '''
# ---- small results archive (manifest). Nothing in this cell may stop the notebook ----
not_finished = ""
try:
    done = [r for r in RUNS if os.path.exists(WORK + "/e6_" + r + ".zip")]
    expected = ["results_" + RUN_NAME + ".zip"] + ["e6_" + r + ".zip" for r in RUNS]
    with zipfile.ZipFile(WORK + "/" + expected[0], "w", zipfile.ZIP_STORED) as z:
        z.writestr("e6/manifest.json", json.dumps({"run": RUN_NAME, "commit": COMMIT, "runs": RUNS, "done": done,
                                                   "failed": failed, "install_level": level, "gpu": gpu,
                                                   "minutes": round((time.time() - t_start) / 60, 1)}, indent=1) + "\\n" + " " * 1200)
    shutil.rmtree(OUT, ignore_errors=True)
    if len(done) < len(RUNS) or failed:
        not_finished = "models not finished: %s; failures: %s" % (sorted(set(RUNS) - set(done)), failed)
except Exception as e:
    not_finished = "packing failed: %s: %s" % (type(e).__name__, e)
RUN = RUN_NAME
print("expected archives:", expected, "|", not_finished or "all models finished")
'''


def main(commit):
    out = ROOT / "finetune" / "notebooks" / "ft_e6_resid.ipynb"
    head = SETUP_HEAD + f'''
RUN_NAME, COMMIT = "ft_e6_resid", "{commit}"
RUNS = {json.dumps(RUNS)}
NEEDS_TOKEN, MODELS, MIN_GPUS = False, "qwen1.5b", 1
EXPECT_PLATFORM = "kaggle"
print("first line of this log:", RUN_NAME, "at commit", COMMIT, "| models:", RUNS)
'''
    cells = [md("# Fine-tuning study, E6: residual stream on POOL-LB\n\nWhat it does: runs the base model and the ten A/B "
                "models (adapters from the attached pair kernels, hash-checked) over POOL-LB and keeps the residual stream "
                "at the final position and the last token of the hypothesis, every layer. Descriptive (PREREG E6); nothing "
                f"is trained.\n\nEstimated time: about 75 minutes on GPU T4 x2. Internet: on. Code: commit `{commit}` of "
                "branch `exp/finetune`. Launched through `finetune/scripts/kaggle_run.py`."),
             code(head), code(FT_PLATFORM_SETUP), code(COMMON_SETUP), code(CLONE), code(ENV), code(DATA), code(ADAPTERS),
             code(RUN), code(PACK), code(FINAL_CHECK)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out.relative_to(ROOT), "| models:", len(RUNS), "| kernel sources:", KERNEL_SOURCES)


if __name__ == "__main__":
    main(sys.argv[1])
