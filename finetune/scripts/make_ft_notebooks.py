"""Write the notebooks of the fine-tuning study, pinned to a pushed commit of exp/finetune.

  ../causal-probe/.venv/bin/python finetune/scripts/make_ft_notebooks.py <commit-sha>

Each notebook is one "Run all": the v1.0 platform cell (output path, GPU check), a clone of the repository at the
pinned commit, the v1.0 pinned environment, the data at pinned revisions, then finetune/scripts/ft_run.py once per
(condition, seed), two at a time (one per T4), and the v1.0 final check that writes STATUS.txt. No Hugging Face
secret is needed: Qwen2.5 is public (and a secret cannot be attached to a notebook pushed through the API).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from notebook_cells import FINAL_CHECK, PLATFORM_SETUP                              # noqa: E402
from make_round2_notebooks import CLONE, COMMON_SETUP, ENV, SETUP_HEAD, code, md     # noqa: E402

S5, S3 = [0, 1, 2, 3, 4], [0, 1, 2]
RUNS = [  # file, run name, jobs (condition, seed), extra args, title, estimate
    dict(file="ft_timetest.ipynb", run="ft_timetest", jobs=[("A", 0)], extra="--time-test",
         title="Fine-tuning study: time test", est="about 20 minutes",
         what="150 training steps of condition A and 300 evaluation items, to measure seconds per step and memory. "
              "No scores are kept and the adapter is discarded (PREREG Section 2)"),
    dict(file="ft_core_A.ipynb", run="ft_core_A", jobs=[("base", 0)] + [("A", s) for s in S5], extra="",
         title="Fine-tuning study: condition A (official), 5 seeds, and the base model's scores", est="set after the time test",
         what="trains five seeds of condition A, scores each and the untuned base model on every evaluation set"),
    dict(file="ft_core_B.ipynb", run="ft_core_B", jobs=[("B", s) for s in S5], extra="",
         title="Fine-tuning study: condition B (balanced), 5 seeds", est="set after the time test",
         what="trains five seeds of condition B and scores each on every evaluation set"),
    dict(file="ft_core_A48.ipynb", run="ft_core_A48", jobs=[("A48", s) for s in S3], extra="",
         title="Fine-tuning study: condition A48 (exposure control), 3 seeds", est="set after the time test", what="three seeds of A48"),
    dict(file="ft_core_Bswap.ipynb", run="ft_core_Bswap", jobs=[("Bswap", s) for s in S3], extra="",
         title="Fine-tuning study: condition Bswap (negative control), 3 seeds", est="set after the time test", what="three seeds of Bswap"),
    dict(file="ft_core_C.ipynb", run="ft_core_C", jobs=[("C", s) for s in S3], extra="",
         title="Fine-tuning study: condition C (premise swapped), 3 seeds", est="set after the time test", what="three seeds of C"),
]

DATA = '''
# ---- data at pinned revisions, and the item sets rebuilt from their stored ids (hash-checked) ----
env = dict(os.environ, PYTHONPATH=REPO_DIR + "/src" + os.pathsep + REPO_DIR + "/finetune", TOKENIZERS_PARALLELISM="false")
for script in ("scripts/download_data.py", "finetune/scripts/download_ft_data.py"):
    r = subprocess.run([PY, script], cwd=REPO_DIR, env=env, capture_output=True, text=True)
    print(r.stdout[-600:] or r.stderr[-1500:])
    assert r.returncode == 0, script + " failed"
r = subprocess.run([PY, "-c", "from ftstudy import datasets as S; import json, pathlib; "
                    "names=sorted(p.stem for p in S.SETS.glob('*.json')); "
                    "print('item sets verified against their hashes:', {n: len(S.materialise(n)) for n in names})"],
                   cwd=REPO_DIR, env=env, capture_output=True, text=True)
print(r.stdout[-1500:] or r.stderr[-2500:])
assert r.returncode == 0, "an item set could not be rebuilt identically: nothing is trained"
print("%.0f min so far" % ((time.time() - t_start) / 60))
'''

RUN = '''
# ---- run: one process per GPU, each working through its share of the jobs (re-running this cell resumes) ----
import threading
OUT = WORK + "/work"
os.makedirs(OUT, exist_ok=True)
n_gpu = max(1, len([l for l in gpu.strip().splitlines()[1:] if l.strip()]))
failed = []
def worker(g, jobs):
    for cond, seed in jobs:
        cmd = [PY, "-u", "finetune/scripts/ft_run.py", "--condition", cond, "--seed", str(seed), "--out", OUT] + EXTRA_ARGS.split()
        p = subprocess.Popen(cmd, cwd=REPO_DIR, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in p.stdout:
            tail = (tail + [line])[-25:]
            if line.startswith("["): print("gpu%d %s" % (g, line), end="")
        if p.wait() != 0:
            failed.append("%s seed %s: exit code %s" % (cond, seed, p.returncode))
            print("gpu%d FAILED %s seed %s\\n%s" % (g, cond, seed, "".join(tail)))
threads = [threading.Thread(target=worker, args=(g, JOBS[g::n_gpu])) for g in range(n_gpu)]
[t.start() for t in threads]; [t.join() for t in threads]
print("failed:", failed or "none", "| total %.0f min" % ((time.time() - t_start) / 60))
'''

PACK = '''
# ---- archives. Nothing in this cell may stop the notebook: the last cell must run whatever happened before ----
import zipfile, glob
expected, skipped, not_finished = ["results_" + RUN_NAME + ".zip"], "", ""
try:
    names = [(c + "_s%d" % s) if c != "base" else "base" for c, s in JOBS]
    missing = [n for n in names if not os.path.exists(OUT + "/" + n + "/DONE")]
    with zipfile.ZipFile(WORK + "/" + expected[0], "w", zipfile.ZIP_STORED) as z:      # stored: the scores are compressed already, and a small archive stays above the final check's 1000 bytes
        for n in names:
            for f in ("run.json", "scores.npz", "log.txt"):
                if os.path.exists(OUT + "/" + n + "/" + f):
                    z.write(OUT + "/" + n + "/" + f, "ft_results/" + RUN_NAME + "/" + n + "/" + f)
        z.writestr("ft_results/" + RUN_NAME + "/manifest.json", json.dumps({"run": RUN_NAME, "commit": COMMIT, "jobs": JOBS, "missing": missing, "failed": failed, "install_level": level}))
    ad = [n for n in names if os.path.isdir(OUT + "/" + n + "/adapter")]
    if ad:
        expected.append("adapters_" + RUN_NAME + ".zip")
        with zipfile.ZipFile(WORK + "/" + expected[1], "w", zipfile.ZIP_STORED) as z:
            for n in ad:
                for f in glob.glob(OUT + "/" + n + "/adapter/*"):
                    z.write(f, "ft_adapters/" + n + "/" + os.path.basename(f))
    shutil.rmtree(OUT, ignore_errors=True)            # the archives are the output; the work folder is not downloaded
    if missing or failed:
        not_finished = "jobs not finished: %s; failures: %s" % (missing, failed)
except Exception as e:
    not_finished = "packing failed: %s: %s" % (type(e).__name__, e)
RUN = RUN_NAME
print("expected archives:", expected, "|", not_finished or "all jobs finished")
'''


def main(commit):
    out = ROOT / "finetune" / "notebooks"
    out.mkdir(parents=True, exist_ok=True)
    for r in RUNS:
        head = (SETUP_HEAD + f'''
RUN_NAME, COMMIT = "{r['run']}", "{commit}"
JOBS = {json.dumps(r['jobs'])}
EXTRA_ARGS = "{r['extra']}"
NEEDS_TOKEN, MODELS, MIN_GPUS = False, "qwen1.5b", 1
print("first line of this log:", RUN_NAME, "at commit", COMMIT, "| jobs:", JOBS)
''')
        cells = [md(f"# {r['title']}\n\nWhat it does: {r['what']}.\n\nEstimated time: {r['est']}. Accelerator: GPU T4 x2. Internet: on. "
                    f"No secret needed. Code: commit `{commit}` of branch `exp/finetune`. Launched through `finetune/scripts/kaggle_run.py`."),
                 code(head), code(PLATFORM_SETUP), code(COMMON_SETUP), code(CLONE), code(ENV), code(DATA), code(RUN), code(PACK), code(FINAL_CHECK)]
        nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        (out / r["file"]).write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", (out / r["file"]).relative_to(ROOT), "| jobs:", len(r["jobs"]))


if __name__ == "__main__":
    main(sys.argv[1])
