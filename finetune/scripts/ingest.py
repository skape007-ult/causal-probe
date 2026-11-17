"""Check a returned run (Kaggle download or the files a Colab run handed to the browser) and file its models.

  ../causal-probe/.venv/bin/python finetune/scripts/ingest.py --dir finetune/kaggle/downloads/ft_core_A1
  ../causal-probe/.venv/bin/python finetune/scripts/ingest.py --dir ~/Downloads/colab_Bswap_s0        # STATUS.txt + the zip files

Refuses the run unless: STATUS.txt exists and its first line is OK; the results archive has a manifest with no
missing or failed job; every model's scores.npz holds every evaluation set with exactly the stored uids in the stored
order, the stored labels, and finite scores; the training set's sha256 in run.json equals the stored set's; the
recipe in run.json is the registered one; the run was not a tiny or time-test run. A model already filed is not
overwritten unless its scores are identical. Files go to finetune/runs/<model>/ (scores.npz, run.json, log.txt) and
adapters to finetune/adapters/<model>/ (not committed). Prints a table; exit code 1 on refusal.
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / "finetune" / "itemsets" / "sets"
RECIPE = dict(rank=16, alpha=32, dropout=0.05, lr=2e-4, warmup=30, batch=16, epochs=3,
              targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
EVAL_SETS = ["eval_pool_lb", "eval_id48", "eval_id", "eval_pool", "eval_official_test", "eval_id48_renamed",
             "eval_id48_renamed_keep_order", "eval_id_renamed", "eval_id_paraphrased", "eval_pool_lb_renamed",
             "eval_authors_refactorization", "eval_authors_paraphrasing"]
SUFFIX = {"rename": "-ren", "rename_keep_order": "-rko", "paraphrase": "-par"}


def check_model(name, z, prefix, problems):
    try:
        info = json.loads(z.read(prefix + "run.json"))
        sc = np.load(io.BytesIO(z.read(prefix + "scores.npz")), allow_pickle=False)
    except KeyError as e:
        problems.append(f"{name}: {e}"); return None
    if info.get("tiny") or info.get("time_test"):
        problems.append(f"{name}: a tiny or time-test run is not a result")
    if info["recipe"] != RECIPE:
        problems.append(f"{name}: recipe differs from the registered one")
    if info["condition"] != "base":
        spec = json.loads((SETS / f"train_{info['condition']}.json").read_text())
        if info["train"]["sha256"] != spec["sha256"] or info["train"]["steps"] != info["train"]["total_steps"]:
            problems.append(f"{name}: trained on a different set, or not to the last step")
    for s in EVAL_SETS:
        spec = json.loads((SETS / f"{s}.json").read_text())
        if s + "__score" not in sc.files:
            problems.append(f"{name}: {s} missing"); continue
        v, uid = sc[s + "__score"], [str(u) for u in sc[s + "__uid"]]
        if len(v) != spec["n"] or not np.isfinite(v).all():
            problems.append(f"{name}: {s} has {len(v)} scores for {spec['n']} items, or a non-finite one")
        if spec["uids"] and uid != [u + SUFFIX.get(spec["transform"], "") for u in spec["uids"]]:
            problems.append(f"{name}: {s} is not the stored items in the stored order")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    problems = []
    st = list(d.rglob("STATUS.txt"))
    if len(st) != 1 or st[0].read_text().splitlines()[:1] != ["OK"]:
        print("REFUSED: STATUS.txt missing, duplicated, or not OK"); sys.exit(1)
    res = [f for f in d.rglob("results_*.zip")]
    if len(res) != 1:
        print("REFUSED: expected exactly one results_*.zip"); sys.exit(1)
    filed = []
    with zipfile.ZipFile(res[0]) as z:
        man = json.loads(z.read(next(n for n in z.namelist() if n.endswith("manifest.json"))))
        if man["missing"] or man["failed"]:
            problems.append(f"manifest: missing {man['missing']}, failed {man['failed']}")
        models = sorted({n.split("/")[2] for n in z.namelist() if n.count("/") == 3})
        for name in models:
            prefix = f"ft_results/{man['run']}/{name}/"
            info = check_model(name, z, prefix, problems)
            if info is None:
                continue
            out = ROOT / "finetune" / "runs" / name
            new = z.read(prefix + "scores.npz")
            if (out / "scores.npz").exists():
                old = np.load(out / "scores.npz"); cur = np.load(io.BytesIO(new))
                if any(not np.array_equal(old[k], cur[k]) for k in cur.files):
                    problems.append(f"{name}: already filed with different scores (a base model scored twice must agree; a seed must not be run twice)")
                continue
            filed.append((name, out, prefix, info, new))
        if problems:
            print("REFUSED:"); [print("  - " + p) for p in problems]; sys.exit(1)
        for name, out, prefix, info, new in filed:
            out.mkdir(parents=True, exist_ok=True)
            (out / "scores.npz").write_bytes(new)
            info["ingested_from"] = {"run": man["run"], "commit": man["commit"], "install_level": man.get("install_level")}
            (out / "run.json").write_text(json.dumps(info, indent=1))
            (out / "log.txt").write_bytes(z.read(prefix + "log.txt"))
            t = info.get("train", {})
            print(f"filed {name:10s} {info['device']} {info.get('gpu')}  train {t.get('seconds_per_step', 0) * t.get('steps', 0) / 60:5.0f} min "
                  f"(last loss {t.get('last_loss')}, max {t.get('max_memory_gb')} GB, oom retries {t.get('out_of_memory_retries')})  "
                  f"scoring {info['eval']['seconds'] / 60:4.0f} min")
    for ad in d.rglob("adapters_*.zip"):
        with zipfile.ZipFile(ad) as z:
            z.extractall(ROOT / "finetune")
    print(f"ok: {len(filed)} model(s) filed from {man['run']} at commit {man['commit'][:7]}")


if __name__ == "__main__":
    main()
