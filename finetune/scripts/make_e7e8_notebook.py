"""Write the E7/E8 power-pilot notebook (PREREG Amendment 3, Section 4), pinned to a pushed commit of exp/finetune.

  ../causal-probe/.venv/bin/python finetune/scripts/make_e7e8_notebook.py <commit-sha>

Same cells as the E6 notebook (platform, clone, pinned environment, data with hash checks, adapters from the attached
pair kernels hash-checked), then finetune/scripts/e7e8_gpu.py for A_s0 and B_s0 on one T4 and A_s1 and B_s1 on the
other. Inputs: finetune/e7e8/pilot_inputs.npz and pilot_prepare.json at the pinned commit (made on the laptop).
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
from make_e6_notebook import ADAPTERS                                              # noqa: E402

RUNS = ["A_s0", "B_s0", "A_s1", "B_s1"]
KERNEL_SOURCES = ["skape007/ft-pair-a0-b0", "skape007/ft-pair-a1-b1"]

RUN = '''
# ---- run: one process per GPU (re-running this cell resumes) ----
import threading, zipfile
OUT = WORK + "/work"
os.makedirs(OUT, exist_ok=True)
n_gpu = max(1, len([l for l in gpu.strip().splitlines()[1:] if l.strip()]))
failed = []
def worker(g, runs):
    cmd = [PY, "-u", "finetune/scripts/e7e8_gpu.py", "--runs", ",".join(runs), "--adapters", AD, "--out", OUT]
    p = subprocess.Popen(cmd, cwd=REPO_DIR, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-25:]
        if line.startswith("["): print("gpu%d %s" % (g, line), end="")
    if p.wait() != 0:
        failed.append("%s: exit code %s" % (runs, p.returncode))
        print("gpu%d FAILED %s\\n%s" % (g, runs, "".join(tail)))
pairs = [RUNS[0::2], RUNS[1::2]] if n_gpu >= 2 else [RUNS]
threads = [threading.Thread(target=worker, args=(g, r)) for g, r in enumerate(pairs)]
[t.start() for t in threads]; [t.join() for t in threads]
print("failed:", failed or "none", "| total %.0f min" % ((time.time() - t_start) / 60))
'''

PACK = '''
# ---- archive. Nothing in this cell may stop the notebook ----
not_finished = ""
try:
    expected = ["results_" + RUN_NAME + ".zip"]
    status = {}
    with zipfile.ZipFile(WORK + "/" + expected[0], "w", zipfile.ZIP_STORED) as z:
        for r in RUNS:
            for f in ("ld.npy", "checks.json", "DONE"):
                if os.path.exists(OUT + "/" + r + "/" + f):
                    z.write(OUT + "/" + r + "/" + f, "e7e8/" + r + "/" + f)
            status[r] = open(OUT + "/" + r + "/DONE").read().strip() if os.path.exists(OUT + "/" + r + "/DONE") else "MISSING"
        z.writestr("e7e8/manifest.json", json.dumps({"run": RUN_NAME, "commit": COMMIT, "status": status, "failed": failed,
                                                     "install_level": level, "minutes": round((time.time() - t_start) / 60, 1)}, indent=1) + "\\n" + " " * 1200)
    if failed or any(v != "OK" for v in status.values()):
        not_finished = "model status: %s; failures: %s" % (status, failed)
except Exception as e:
    not_finished = "packing failed: %s: %s" % (type(e).__name__, e)
RUN = RUN_NAME
print("expected archives:", expected, "|", not_finished or "all models OK")
'''


def main(commit):
    out = ROOT / "finetune" / "notebooks" / "ft_e7e8_pilot.ipynb"
    head = SETUP_HEAD + f'''
RUN_NAME, COMMIT = "ft_e7e8_pilot", "{commit}"
RUNS = {json.dumps(RUNS)}
NEEDS_TOKEN, MODELS, MIN_GPUS = False, "qwen1.5b", 1
EXPECT_PLATFORM = "kaggle"
print("first line of this log:", RUN_NAME, "at commit", COMMIT, "| models:", RUNS)
'''
    cells = [md("# Fine-tuning study, E7/E8 power pilot (PREREG Amendment 3, Section 4)\n\nWhat it does: steering and "
                "patching at the registered site for A_s0, A_s1, B_s0, B_s1 on 200 pilot items, after the Section 5 checks. "
                "Its numbers set the item count of the registered runs; it is not a test.\n\nEstimated time: about 30 "
                f"minutes on GPU T4 x2. Internet: on. Code: commit `{commit}` of branch `exp/finetune`."),
             code(head), code(FT_PLATFORM_SETUP), code(COMMON_SETUP), code(CLONE), code(ENV), code(DATA), code(ADAPTERS),
             code(RUN), code(PACK), code(FINAL_CHECK)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print("wrote", out.relative_to(ROOT), "| models:", RUNS, "| kernel sources:", KERNEL_SOURCES)


if __name__ == "__main__":
    main(sys.argv[1])
