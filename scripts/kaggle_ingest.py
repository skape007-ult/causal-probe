"""Validate the files downloaded from the Kaggle run (zips, or the folders a browser made by
unzipping them) and fold them into this repository.

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


def pinned_commit(run: str | None = None) -> str | None:
    """The commit pinned in the notebook of this run (round 1: notebooks/kaggle_gpu.ipynb; round 2:
    the notebook whose RUN_NAME is `run`)."""
    nbs = [ROOT / "notebooks" / "kaggle_gpu.ipynb"] if run is None else sorted((ROOT / "notebooks").glob("*_r2_*.ipynb"))
    for nb in nbs:
        txt = nb.read_text() if nb.exists() else ""
        if run is not None and not re.search(r'RUN_NAME\s*=\s*\\"%s\\"' % re.escape(run), txt):
            continue
        m = re.search(r'COMMIT\s*=\s*\\"([0-9a-f]{40})', txt)
        if m:
            return m.group(1)
    return None


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


class Src:
    """One downloaded archive: a zip file, or the folder a browser made by unzipping it
    (Safari unpacks downloads; several cache zips then become cache/, cache-1/, 'cache 2'/ ...).
    Member names are the names inside the original zip ('results/...', 'cache/<model>/...')."""

    def __init__(self, path: Path):
        self.path, self.is_zip = path, path.is_file()
        if self.is_zip:
            self.z = zipfile.ZipFile(path)
            self.names = [n for n in self.z.namelist() if not n.endswith("/")]
            self.top = "results" if path.name.startswith("results") else "cache"
        else:
            self.top = "results" if path.name.startswith("results") else "cache"
            self.names = sorted(f"{self.top}/{f.relative_to(path).as_posix()}" for f in path.rglob("*")
                                if f.is_file() and f.name != ".DS_Store")

    @property
    def label(self) -> str:
        return self.path.name + ("" if self.is_zip else "/ (already unzipped)")

    def size(self) -> int:
        return self.path.stat().st_size if self.is_zip else sum((self.path / Path(n).relative_to(self.top)).stat().st_size for n in self.names)

    def test(self):
        return self.z.testzip() if self.is_zip else None

    def read(self, name: str) -> bytes:
        return self.z.read(name) if self.is_zip else (self.path / Path(name).relative_to(self.top)).read_bytes()

    def copy_to(self, name: str, dst: Path):
        import shutil
        if self.is_zip:
            with self.z.open(name) as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 24)
        else:
            shutil.copyfile(self.path / Path(name).relative_to(self.top), dst)


def find_sources(src: Path):
    res = sorted(src.glob("results*.zip")) + sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("results"))
    caches = sorted(src.glob("cache*.zip")) + sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("cache"))
    return (Src(res[0]) if res else None), [Src(c) for c in caches]


def git_blob(commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="round-2 run name (r2_qwen3b, r2_qwen7b, r2_gemma, r2_qwen15b); "
                                                "the downloads are then expected in kaggle_in/<run>/")
    ap.add_argument("--dir", default=None, help="folder with the downloads (default: kaggle_in/, or kaggle_in/<run>/ with --run)")
    ap.add_argument("--commit", default=None, help="expected code commit (default: the one pinned in the notebook)")
    ap.add_argument("--large", default="qwen3b", help="larger model that was run; 'none' if LARGE_MODEL was None")
    ap.add_argument("--no-gemma", action="store_true", help="the run had RUN_GEMMA = False")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fold", action="store_true", help="extract only; do not regenerate tables/figures/documents")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from causal_probe.config import CACHE, QUANT_4BIT, RESULTS

    src = Path(a.dir) if a.dir else (ROOT / "kaggle_in" / a.run if a.run else ROOT / "kaggle_in")
    large = None if a.large.lower() == "none" else a.large
    commit = a.commit or pinned_commit(a.run)
    problems, notes = [], []
    report = {"dir": str(src), "expected_commit": commit}

    # ------------------------------------------------------------------ 1. zips
    R, caches = find_sources(src) if src.is_dir() else (None, [])
    if R is None:
        sys.exit(f"ERROR: neither results.zip nor an unzipped results/ folder in {src}/. Put the downloaded "
                 f"results.zip and cache_*.zip there (zipped or unzipped) and re-run.")
    print("== 1. files ==")
    for S in [R] + caches:
        bad = S.test()
        print(f"  {S.label:38s} {S.size() / 1e6:9.1f} MB  {len(S.names):4d} files  {'CORRUPT at ' + bad if bad else 'ok'}")
        if bad:
            problems.append(f"{S.label} is corrupt (first bad member: {bad}); download it again")
    if problems:
        sys.exit("ERROR: " + "; ".join(problems))
    names = set(R.names)

    def zjson(name):
        return json.loads(R.read(name)) if name in names else None

    # ------------------------------------------------------------------ 2. commit
    print("== 2. code version ==")
    env = zjson(f"results/round2_environment_{a.run}.json" if a.run else "results/kaggle_environment.json")
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
    status = zjson(f"results/round2_status_{a.run}.json" if a.run else "results/kaggle_run_status.json") or {"stages": {}}
    st = status["stages"]
    if a.run:
        exp = status.get("planned", [])
        if not exp:
            problems.append(f"results/round2_status_{a.run}.json is missing or lists no planned stages (wrong folder, or the run never started)")
    else:
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
    if a.run:
        exp_models = list(status.get("models_run", []))
    else:
        exp_models = (["gemma2b", "gemma2b-rand"] if not a.no_gemma else []) + \
                     ([large] + ([] if large in QUANT_4BIT else [f"{large}-rand"]) if large else [])
    have = {Path(n).parts[1] for S in caches for n in S.names}          # model folders present in any cache archive
    for mn in exp_models:
        if mn not in have:
            (notes if mn.endswith("-rand") else missing).append(f"cache_{mn}.zip (not in {src.name}/)")
            print(f"  {'note' if mn.endswith('-rand') else 'MISSING'}: cache_{mn}.zip is not in {src.name}/"
                  + (" (optional: random-init control)" if mn.endswith("-rand") else ""))
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
        if a.run and (n in ("results/kaggle_environment.json", "results/kaggle_run_status.json", "results/kaggle_ingest_report.json")
                      or n.startswith("results/kaggle_logs/")):
            continue                           # round-1 records; this run's own are round2_*_<run>
        blob = R.read(n)
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
    for S in caches:
        dirs = sorted({str(Path(n).parent) for n in S.names})
        for d in dirs:                      # d = cache/<model>[-rand]/<tag>  (or cache/<model> for patching.npz)
            rel = Path(d).relative_to("cache")
            dst = CACHE / rel
            members = [n for n in S.names if str(Path(n).parent) == d]
            zip_has_resid = any(n.endswith("resid.npy") for n in members)
            local_has_resid = (dst / "resid.npy").exists()
            dst.mkdir(parents=True, exist_ok=True)
            core = ("meta.json", "readout.npz", "resid.npy")
            mode = "all"                    # copy everything
            if local_has_resid and zip_has_resid:
                # replace the local activations only if the download holds strictly more token
                # positions for the same items (round 2 adds the end-of-hypothesis token)
                lm = json.loads((dst / "meta.json").read_text())
                im = json.loads(S.read(next(n for n in members if n.endswith("meta.json"))))
                more = set(im["positions"]) > set(lm["positions"]) and im["uids"] == lm["uids"]
                mode = "all" if more else "extras"
            elif local_has_resid and not zip_has_resid:
                mode = "beside"             # keep the laptop cache; store the GPU run's logits beside it
            for n in members:
                base = Path(n).name
                if base in core:
                    if mode == "extras":
                        continue
                    if mode == "beside":
                        base = {"meta.json": "meta_kaggle.json", "readout.npz": "readout_kaggle.npz"}[base]
                S.copy_to(n, dst / base)
            kind = {"all": "activations + logits" if zip_has_resid else "logits / generated answers / raw intervention outputs",
                    "extras": "raw intervention outputs only (local activations kept)",
                    "beside": "logits beside the local cache, raw intervention outputs"}[mode]
            print(f"  {str(rel):38s} {kind}")
            if zip_has_resid and mode == "all":
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
    (RESULTS / (f"round2_ingest_report_{a.run}.json" if a.run else "kaggle_ingest_report.json")).write_text(json.dumps(report, indent=1))

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
        shown = [m for m in (status.get("models_run", []) if a.run else [large]) if m and "behaviour" in (status.get("jobs") or ["behaviour"])]
        for large in [m for m in shown if ab.exists() and m in json.loads(ab.read_text())]:
            print(f"\n== behaviour of {large} (logit-scored; is it above chance?) ==")
            for tag, v in json.loads(ab.read_text())[large].items():
                print(f"  {tag:22s} {v['mode']:28s} Yes-rate {v['yes_rate']:.2f}  accuracy {v['accuracy']['est']:.3f} "
                      f"(majority {v['majority']:.3f})  AUROC {v['auroc']['est']:.3f} [{v['auroc']['lo']:.3f}, {v['auroc']['hi']:.3f}]  "
                      f"best-threshold acc {v['best_threshold_acc']['est']:.3f}  -> {v['reading']}")
    print("\n" + ("DONE with problems:\n  - " + "\n  - ".join(problems) if problems else "DONE: everything expected is present and consistent."))
    if not a.run:
        print("Next (see notes/progress.md, 'When the Kaggle zips come back'): ./scripts/guarded.sh tier1 ./scripts/run_tier1.sh "
              + (large or "<model>") + " 2    (probes and baselines for the new model, CPU, a few hours)")
    sys.exit(1 if problems and not a.allow_partial else 0)


if __name__ == "__main__":
    main()
