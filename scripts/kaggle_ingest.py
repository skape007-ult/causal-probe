"""Validate the zip files downloaded from the Kaggle run and fold them into this repository.

  .venv/bin/python scripts/kaggle_ingest.py                  # zips in ./kaggle_in/
  .venv/bin/python scripts/kaggle_ingest.py --allow-partial  # accept a run that did not finish
  .venv/bin/python scripts/kaggle_ingest.py --dry-run        # validate and report only, change nothing

What it does, in order (see notes/kaggle_ingest.md):
  1. Checks every zip is readable and lists what is in it.
  2. Checks the run used the pinned code commit (results/kaggle_environment.json -> git_commit).
  3. Checks every expected stage is present and succeeded (results/kaggle_run_status.json).
     Stops here with exit code 1 if something is missing or failed, unless --allow-partial.
  4. Extracts results.zip, but only the files the Kaggle run created or changed. The zip holds
     Kaggle's whole results/ folder as of the pinned commit, including older copies of files that
     have since been updated locally; those are compared with the pinned commit and skipped.
  5. Extracts the cache zips. A cache folder that already has local activations (Gemma's original
     item sets) is never overwritten: Kaggle's per-item logits for it are stored next to the local
     files as meta_kaggle.json / readout_kaggle.npz, and only steering*.npz is added.
  6. Checks every extracted cache loads and matches the local item lists.
  7. Regenerates the answer-bias report, tables, figures and documents, and prints the larger
     model's behaviour on the original and debiased item sets.
It does NOT run the probe analysis for the new model (hours of CPU): `scripts/run_tier1.sh <model> 2`.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def pinned_commit() -> str | None:
    nb = ROOT / "notebooks" / "kaggle_gpu.ipynb"
    m = re.search(r'COMMIT\s*=\s*\\"([0-9a-f]{40})', nb.read_text()) if nb.exists() else None
    return m.group(1) if m else None


def expected_stages(gemma: bool, large: str | None, quant: bool) -> list[str]:
    """Stage names of scripts/kaggle_run.py at the pinned commit (full run, not --tiny)."""
    out = ["data", "environment"]
    if gemma:
        g = "gemma.gemma2b"
        out += [f"{g}.cache_c2c", f"{g}.cache", f"{g}.cache_random", f"{g}.cache_agreement"]
        out += [f"{g}.steer.{t}" for t in ("c2c_probe", "cladder", "c2c_debiased")]
        out += [f"{g}.steer.{t}.d{d}" for t in ("c2c_probe", "cladder") for d in ("0.5", "0.85")]
    if large:
        m = f"large.{large}"
        out += [f"{m}.cache", f"{m}.behaviour", f"{m}.lens"]
        out += [f"{m}.steer.{t}" for t in ("c2c_probe", "cladder", "c2c_debiased")]
        out += [f"{m}.steer.c2c_probe.d{d}" for d in ("0.5", "0.85")]
        if not quant:
            out.append(f"{m}.cache_random")
    if gemma:
        out += ["gemma.gemma2b.patching", "gemma.gemma2b.sae_features", "gemma.gemma2b.sae_ablate"]
    return out


def git_blob(commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "kaggle_in"), help="folder with results.zip and cache_*.zip")
    ap.add_argument("--commit", default=None, help="expected code commit (default: the one pinned in the notebook)")
    ap.add_argument("--large", default="qwen3b", help="larger model that was run; 'none' if LARGE_MODEL was None")
    ap.add_argument("--no-gemma", action="store_true", help="the run had RUN_GEMMA = False")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fold", action="store_true", help="extract only; do not regenerate tables/figures/documents")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from causal_probe.config import CACHE, QUANT_4BIT, RESULTS

    src = Path(a.dir)
    large = None if a.large.lower() == "none" else a.large
    commit = a.commit or pinned_commit()
    problems, notes = [], []
    report = {"dir": str(src), "expected_commit": commit}

    # ------------------------------------------------------------------ 1. zips
    rz = src / "results.zip"
    if not rz.exists():
        sys.exit(f"ERROR: {rz} not found. Put the downloaded results.zip (and cache_*.zip) into {src}/ and re-run.")
    zips = [rz] + sorted(src.glob("cache_*.zip"))
    print("== 1. files ==")
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            bad = z.testzip()
            names = z.namelist()
            print(f"  {zp.name:28s} {zp.stat().st_size / 1e6:9.1f} MB  {len(names):4d} files  {'CORRUPT at ' + bad if bad else 'ok'}")
            if bad:
                problems.append(f"{zp.name} is corrupt (first bad member: {bad}); download it again")
    if problems:
        sys.exit("ERROR: " + "; ".join(problems))
    Z = zipfile.ZipFile(rz)
    names = set(Z.namelist())

    def zjson(name):
        return json.loads(Z.read(name)) if name in names else None

    # ------------------------------------------------------------------ 2. commit
    print("== 2. code version ==")
    env = zjson("results/kaggle_environment.json")
    if env is None:
        problems.append("results/kaggle_environment.json is missing (the run stopped before its second stage)")
    else:
        got = env.get("git_commit") or ""
        okc = bool(commit) and got.startswith(commit[:7]) and commit.startswith(got[:7])
        print(f"  run used commit {got[:12] or '?'}; expected {str(commit)[:12]}  -> {'ok' if okc else 'MISMATCH'}")
        print(f"  torch {env.get('torch')}, transformers {env.get('transformers')}, GPU {env.get('gpus')}")
        if not okc:
            problems.append(f"code commit mismatch: run used {got[:12]}, expected {str(commit)[:12]}")
        report["environment"] = env

    # ------------------------------------------------------------------ 3. stages
    print("== 3. stages ==")
    status = zjson("results/kaggle_run_status.json") or {"stages": {}}
    st = status["stages"]
    exp = expected_stages(not a.no_gemma, large, bool(large) and large in QUANT_4BIT)
    missing, failed = [], []
    for name in exp:
        s = st.get(name)
        if s is None and name.endswith(".cache") and st.get(name + "_resume", {}).get("ok"):
            s = st[name + "_resume"]
        if s is None:
            missing.append(name); print(f"  MISSING              {name}")
        elif not s["ok"]:
            failed.append(name); print(f"  FAILED  {s['minutes']:6.1f} min  {name}   <- {s['detail']}")
            for line in s.get("log_tail", [])[-3:]:
                print(f"            | {line[:160]}")
        else:
            print(f"  ok      {s['minutes']:6.1f} min  {name}")
    finished = "export" in status
    print(f"  run reached the export step: {'yes' if finished else 'NO (the run was cut short)'}; "
          f"total {status.get('total_minutes', '?')} min")
    exp_zips = (["cache_gemma2b.zip", "cache_gemma2b-rand.zip"] if not a.no_gemma else []) + \
               ([f"cache_{large}.zip"] + ([] if large in QUANT_4BIT else [f"cache_{large}-rand.zip"]) if large else [])
    have = {z.name for z in zips}
    for zn in exp_zips:
        if zn not in have:
            (notes if zn.endswith("-rand.zip") else missing).append(f"{zn} (not in {src.name}/)")
            print(f"  {'note' if zn.endswith('-rand.zip') else 'MISSING'}: {zn} is not in {src.name}/"
                  + (" (optional: random-init control)" if zn.endswith("-rand.zip") else ""))
    report.update(missing=missing, failed=failed, run_finished=finished, notes=notes)
    if missing or failed or not finished:
        problems.append(f"{len(missing)} missing and {len(failed)} failed stage(s)"
                        + ("" if finished else "; run did not reach the export step"))
    if problems:
        print("\nPROBLEMS:\n  - " + "\n  - ".join(problems))
        if not a.allow_partial:
            print("\nNothing was changed. Re-run with --allow-partial to ingest what is there "
                  "(see notes/kaggle_ingest.md, 'Partial runs').")
            sys.exit(1)
    if a.dry_run:
        print("\n--dry-run: validation only, nothing extracted.")
        sys.exit(1 if problems else 0)

    # ------------------------------------------------------------------ 4. results
    print("== 4. extracting results.zip (only files the run created or changed) ==")
    new, skipped, conflicts = [], 0, []
    for n in sorted(names):
        if n.endswith("/") or not n.startswith("results/"):
            continue
        blob = Z.read(n)
        if commit and git_blob(commit, n) == blob:
            skipped += 1                       # unchanged copy of a file from the pinned commit
            continue
        dst = RESULTS / Path(n).relative_to("results")
        if dst.exists() and dst.read_bytes() != blob and git_blob(commit, n) is not None and git_blob("HEAD", n) != git_blob(commit, n):
            conflicts.append(n)                # changed both by the run and locally since the pin
            dst = dst.with_name(dst.name + ".kaggle")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)
        new.append(n)
    print(f"  wrote {len(new)} files, skipped {skipped} unchanged")
    for c in conflicts:
        print(f"  CONFLICT: {c} was changed both on Kaggle and locally; Kaggle's copy saved as {c}.kaggle")
    report.update(results_written=new, conflicts=conflicts)

    # ------------------------------------------------------------------ 5. caches
    print("== 5. extracting caches ==")
    loaded = []
    for zp in zips[1:]:
        with zipfile.ZipFile(zp) as z:
            dirs = sorted({str(Path(n).parent) for n in z.namelist() if not n.endswith("/")})
            for d in dirs:                      # d = cache/<model>[-rand]/<tag>  (or cache/<model> for patching.npz)
                rel = Path(d).relative_to("cache")
                dst = CACHE / rel
                members = [n for n in z.namelist() if str(Path(n).parent) == d and not n.endswith("/")]
                zip_has_resid = any(n.endswith("resid.npy") for n in members)
                local_has_resid = (dst / "resid.npy").exists()
                dst.mkdir(parents=True, exist_ok=True)
                for n in members:
                    base = Path(n).name
                    if local_has_resid and not zip_has_resid:
                        # keep the laptop cache intact; store Kaggle's logits beside it
                        if base == "meta.json":
                            base = "meta_kaggle.json"
                        elif base == "readout.npz":
                            base = "readout_kaggle.npz"
                    elif local_has_resid and zip_has_resid:
                        print(f"  SKIPPED {rel}: activations exist both locally and in the zip (delete the local folder to replace it)")
                        break
                    (dst / base).write_bytes(z.read(n))
                else:
                    kind = "activations + logits" if zip_has_resid else ("logits only, beside the local cache" if local_has_resid else "logits / generated answers only")
                    print(f"  {str(rel):38s} {kind}")
                    if zip_has_resid:
                        loaded.append(rel)

    # ------------------------------------------------------------------ 6. caches load
    print("== 6. checking extracted caches ==")
    import numpy as np
    from causal_probe import cache as C
    for rel in loaded:
        name, tag = rel.parts[0], rel.parts[1]
        rand = name.endswith("-rand")
        model = name.removesuffix("-rand")
        try:
            resid, meta, items = C.load_cache(model, tag, random_init=rand)      # asserts item order
            assert list(resid.shape) == meta["shape"], (resid.shape, meta["shape"])
            assert np.isfinite(np.asarray(resid[:: max(1, len(resid) // 50)], dtype=np.float32)).all()
            print(f"  ok   {str(rel):38s} {tuple(resid.shape)}  positions {meta['positions']}")
        except Exception as e:
            problems.append(f"cache {rel} does not load: {type(e).__name__}: {e}")
            print(f"  FAIL {rel}: {type(e).__name__}: {e}")
    report["caches_loaded"] = [str(r) for r in loaded]
    report["problems"] = problems
    (RESULTS / "kaggle_ingest_report.json").write_text(json.dumps(report, indent=1))

    # ------------------------------------------------------------------ 7. fold in
    if not a.no_fold:
        print("== 7. regenerating answer-bias report, tables, figures, documents ==")
        import os
        env2 = dict(os.environ, OMP_NUM_THREADS="2", PYTHONPATH=str(ROOT / "src"))
        for script in ("15_answer_bias.py", "12_tables.py", "06_figures.py", "13_build_docs.py"):
            r = subprocess.run([PY, f"scripts/{script}"], cwd=ROOT, env=env2, capture_output=True, text=True)
            print(f"  {script:22s} {'ok' if r.returncode == 0 else 'FAILED'}")
            if r.returncode != 0:
                problems.append(f"{script} failed")
                print("    " + "\n    ".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
        ab = RESULTS / "answer_bias.json"
        if large and ab.exists() and large in json.loads(ab.read_text()):
            print(f"\n== behaviour of {large} (logit-scored; is it above chance?) ==")
            for tag, v in json.loads(ab.read_text())[large].items():
                print(f"  {tag:22s} {v['mode']:28s} Yes-rate {v['yes_rate']:.2f}  accuracy {v['accuracy']['est']:.3f} "
                      f"(majority {v['majority']:.3f})  AUROC {v['auroc']['est']:.3f} [{v['auroc']['lo']:.3f}, {v['auroc']['hi']:.3f}]  "
                      f"best-threshold acc {v['best_threshold_acc']['est']:.3f}  -> {v['reading']}")
    print("\n" + ("DONE with problems:\n  - " + "\n  - ".join(problems) if problems else "DONE: everything expected is present and consistent."))
    print("Next (see notes/progress.md, 'When the Kaggle zips come back'): ./scripts/guarded.sh tier1 ./scripts/run_tier1.sh "
          + (large or "<model>") + " 2    (probes and baselines for the new model, CPU, a few hours)")
    sys.exit(1 if problems and not a.allow_partial else 0)


if __name__ == "__main__":
    main()
