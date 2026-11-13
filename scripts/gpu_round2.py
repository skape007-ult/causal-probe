"""Round-2 GPU jobs (free Kaggle or Colab GPU; see KAGGLE.md, "Round 2"). Also runs anywhere else.

Same machinery as scripts/kaggle_run.py: every stage is one subprocess, is skipped if its marker
file exists (re-running resumes), has a time limit, and may fail without stopping the stages after
it. The status of every stage goes to results/round2_status_<run>.json; the small results archive is
refreshed after every stage.

  python scripts/gpu_round2.py --run r2_qwen3b --models qwen3b \
      --jobs cache,controls,controls_hyp,patch,patch_same,feature,export --out /kaggle/working
  python scripts/gpu_round2.py --run test --models qwen0.5b --jobs ... --tiny --out <dir>     # local smoke test

Jobs (each applies to every model in --models, in this order):
  behaviour     per-item answer logits on the debiased pool, the official Corr2Cause test set, CLadder
                and the original probe pool (in that order, resumable), then the answer-bias report.
                For a model that has no behaviour numbers yet (the 7B check).
  cache         activations for c2c_probe and c2c_debiased (what the jobs below need)
  controls      matched steering controls at the final token (20 shuffled-label directions of each
                kind, 100 isotropic): scripts/07c_steering_controls.py
  controls_hyp  the same at the last token of the hypothesis
  patch         activation patching on label-changing minimal pairs (scripts/08_patching.py), for
                models that have no patching result yet
  patch_same    the same on label-preserving pairs (the control for "patching localises a prompt
                difference")
  feature       interventions on the adjacency direction with matched controls, at four sites
                (final token / end of hypothesis x best layer / first layer at 90% of the best):
                scripts/16_feature_causal.py
  export        cache_<run>_<model>.zip: debiased-pool activations at the final and end-of-hypothesis
                tokens, per-item logits, and the raw logit differences of every intervention
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
T0 = time.time()
JOBS = ["behaviour", "cache", "controls", "controls_hyp", "patch", "patch_same", "feature", "export"]
SITES = [("final", "best"), ("content_end", "best"), ("final", "first90"), ("content_end", "first90")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="name of this run (in the output file names)")
    ap.add_argument("--models", required=True, help="comma-separated model names from config")
    ap.add_argument("--jobs", required=True, help="comma-separated subset of: " + ",".join(JOBS))
    ap.add_argument("--fallback", default="", help="model:fallback pairs, e.g. qwen7b-4bit:qwen7b - used if the first cache stage of 'model' fails")
    ap.add_argument("--out", default=str(ROOT / "kaggle_out"))
    ap.add_argument("--tiny", action="store_true", help="smoke test: a few items per stage")
    ap.add_argument("--skip", default="", help="comma-separated stage-name prefixes to skip")
    ap.add_argument("--stage-minutes", type=float, default=60.0)
    ap.add_argument("--export-positions", default="final,content_end")
    a = ap.parse_args()
    jobs = [j for j in a.jobs.split(",") if j]
    assert all(j in JOBS for j in jobs), jobs
    fallback = dict(p.split(":") for p in a.fallback.split(",") if p)

    sys.path.insert(0, str(ROOT / "src"))
    from causal_probe import cache as C
    from causal_probe.config import CACHE, RESULTS

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    logs = RESULTS / f"round2_logs_{a.run}"; logs.mkdir(exist_ok=True)
    marks = CACHE / f"_round2_markers_{a.run}"; marks.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS / f"round2_status_{a.run}.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    status.update(run=a.run, models=a.models.split(","), jobs=jobs, tiny=a.tiny, planned=[])
    skip = [s for s in a.skip.split(",") if s]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               TOKENIZERS_PARALLELISM="false")

    def zip_small():
        with zipfile.ZipFile(out / f"results_{a.run}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(RESULTS.rglob("*")):
                if f.is_file() and f.stat().st_size < 20e6:
                    z.write(f, Path("results") / f.relative_to(RESULTS))

    def stage(name: str, *cmd: str, minutes: float | None = None, optional: bool = False) -> bool:
        if not optional:
            status["planned"].append(name)
        if any(name.startswith(s) for s in skip):
            print(f"[skip] {name} (--skip)", flush=True)
            return False
        if (marks / name).exists():
            print(f"[done] {name} (marker found)", flush=True)
            return True
        t = time.time()
        print(f"\n[start] {name}  (+{(t - T0) / 60:.0f} min)\n        {' '.join(cmd)}", flush=True)
        log = logs / f"{name}.log"
        try:
            with open(log, "a") as lf:
                p = subprocess.run([PY, *cmd], cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=(minutes or a.stage_minutes) * 60)
            ok, why = p.returncode == 0, f"exit code {p.returncode}"
        except subprocess.TimeoutExpired:
            ok, why = False, f"timeout after {(minutes or a.stage_minutes):.0f} min"
        mins = (time.time() - t) / 60
        tail = log.read_text(errors="replace").strip().splitlines()[-8:]
        status["stages"][name] = {"ok": ok, "minutes": round(mins, 1), "detail": "ok" if ok else why, "log_tail": tail}
        status["total_minutes"] = round((time.time() - T0) / 60, 1)
        status_path.write_text(json.dumps(status, indent=1))
        print(f"[{'ok' if ok else 'FAILED: ' + why}] {name}  {mins:.1f} min", flush=True)
        for line in tail:
            print("        | " + line[:200], flush=True)
        if ok:
            (marks / name).write_text("done")
        zip_small()
        return ok

    stage("data", "scripts/download_data.py", minutes=30)
    if stage("environment", "scripts/kaggle_env.py", minutes=10):
        src = RESULTS / "kaggle_environment.json"
        if src.exists():
            shutil.copyfile(src, RESULTS / f"round2_environment_{a.run}.json")

    # the smoke test builds the two caches in full (the selection/report split needs whole blocks);
    # only the number of items each intervention stage runs is reduced
    models_run = []
    for M0 in a.models.split(","):
        M = M0
        # ------------------------------------------------------------ behaviour (new model)
        if "behaviour" in jobs:
            tags = "c2c_debiased,c2c_test,cladder,c2c_probe"

            def beh(m, k):
                return stage(f"r2.{m}.behaviour.cache_part{k}", "scripts/01_build_cache.py", "--model", m, "--tags", tags, "--no-gen",
                             "--max-minutes", "50", *([] if not a.tiny else ["--limit", "64"]), minutes=58, optional=True)
            ok = beh(M, 0)
            if not ok and M in fallback and status["stages"].get(f"r2.{M}.behaviour.cache_part0", {}).get("minutes", 99) < 12:
                print(f"[fallback] {M} failed early; trying {fallback[M]}", flush=True)
                M = fallback[M]
                ok = beh(M, 0)
            k = 1
            while not ok and k < 4 and not a.tiny:        # the cache builder stops itself after 50 min and resumes on re-run
                ok = beh(M, k); k += 1
            # one planned entry for the whole (resumable) caching step
            parts = [v for n_, v in status["stages"].items() if n_.startswith(f"r2.{M}.behaviour.cache_part")]
            status["planned"].append(f"r2.{M}.behaviour.cache")
            status["stages"][f"r2.{M}.behaviour.cache"] = {"ok": bool(ok), "minutes": round(sum(v["minutes"] for v in parts), 1),
                                                           "detail": "ok" if ok else "not finished after %d parts" % len(parts),
                                                           "log_tail": parts[-1]["log_tail"] if parts else []}
            status_path.write_text(json.dumps(status, indent=1))
            # report whatever was cached, also after a timeout (the debiased pool comes first)
            stage(f"r2.{M}.behaviour.report", "scripts/15_answer_bias.py", minutes=10)
        # ------------------------------------------------------------ cache for the interventions
        have_cache = True
        if "cache" in jobs:
            have_cache = stage(f"r2.{M}.cache", "scripts/01_build_cache.py", "--model", M, "--tags", "c2c_probe,c2c_debiased",
                               "--no-gen", "--max-minutes", "50", minutes=58)
            if not have_cache and M in fallback and status["stages"][f"r2.{M}.cache"]["minutes"] < 12:
                M = fallback[M]
                have_cache = stage(f"r2.{M}.cache", "scripts/01_build_cache.py", "--model", M, "--tags", "c2c_probe,c2c_debiased",
                                   "--no-gen", "--max-minutes", "50", minutes=58)
        models_run.append(M)
        n_c = ["--n", "10"] if a.tiny else ["--n", "150"]
        stride = ["--select-stride", "6" if a.tiny else "2"]
        if have_cache:
            if "controls" in jobs:
                for tag in ("c2c_debiased", "c2c_probe"):
                    stage(f"r2.{M}.controls.{tag}", "scripts/07c_steering_controls.py", "--model", M, "--tag", tag,
                          "--sub-batch", "16", *stride, *n_c, minutes=40)
            if "controls_hyp" in jobs:
                for tag in ("c2c_debiased", "c2c_probe"):
                    stage(f"r2.{M}.controls_hyp.{tag}", "scripts/07c_steering_controls.py", "--model", M, "--tag", tag,
                          "--position", "content_end", "--sub-batch", "16", "--max-minutes", "50", *stride, *n_c, minutes=58)
            if "feature" in jobs:
                for pos, spec in SITES:
                    stage(f"r2.{M}.feature.{pos}.{spec}", "scripts/16_feature_causal.py", "--model", M, "--position", pos,
                          "--layer-spec", spec, "--sub-batch", "16", "--max-minutes", "45",
                          *(["--per-template", "4", "--n-shuf", "3", "--n-iso", "2", "--select-stride", "6"] if a.tiny
                            else ["--per-template", "100", "--select-stride", "1"]), minutes=70)   # fitting ~130 probes on CPU comes first
        pp = ["--n-pairs", "6", "--layer-stride", "6", "--batch", "4"] if a.tiny else ["--n-pairs", "60", "--layer-stride", "2", "--batch", "8"]
        if "patch_same" in jobs:
            stage(f"r2.{M}.patch_same", "scripts/08_patching.py", "--model", M, "--label-mode", "same", "--suffix", "_same",
                  "--skip-components", *pp, minutes=40)
        if "patch" in jobs:
            stage(f"r2.{M}.patch", "scripts/08_patching.py", "--model", M, *pp,
                  *(["--n-comp-pairs", "4"] if a.tiny else ["--n-comp-pairs", "20"]), minutes=50)

    # ------------------------------------------------------------------ export
    sizes, zsizes = {}, {}
    exp = out / "export"
    pos = tuple(a.export_positions.split(","))
    for M in models_run:
        for tag in ("c2c_debiased", "c2c_probe", "c2c_test", "cladder"):
            d = C.cache_dir(M, tag)
            if not (d / "meta.json").exists():
                continue
            full = tag == "c2c_debiased" and "export" in jobs
            meta = json.loads((d / "meta.json").read_text())
            sizes[f"{M}/{tag}"] = C.export_cache(M, tag, exp, False, tuple(p for p in pos if p in meta["positions"]) if full else ("final",),
                                                 with_resid=full)
        for f in (CACHE / M).glob("patching*.npz"):
            (exp / M).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, exp / M / f.name)
    if exp.exists():
        for mdir in sorted(p for p in exp.iterdir() if p.is_dir()):
            zp = out / f"cache_{a.run}_{mdir.name}.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
                for f in sorted(mdir.rglob("*")):
                    if f.is_file():
                        z.write(f, Path("cache") / mdir.name / f.relative_to(mdir))
            zsizes[zp.name] = round(zp.stat().st_size / 1e6, 1)
        shutil.rmtree(exp)
    status["planned"] = list(dict.fromkeys(status["planned"]))
    status["models_run"] = models_run
    status["export"] = {"positions": list(pos), "bytes_per_cache_dir": sizes, "zip_megabytes": zsizes}
    status["total_minutes"] = round((time.time() - T0) / 60, 1)
    status["all_ok"] = all(status["stages"].get(n, {}).get("ok", False) for n in status["planned"])
    status_path.write_text(json.dumps(status, indent=1))
    zip_small()
    print("\n================ summary ================")
    for k in status["planned"]:
        v = status["stages"].get(k)
        print("not run        " + k if v is None else f"{'ok    ' if v['ok'] else 'FAILED'} {v['minutes']:6.1f} min  {k}  {'' if v['ok'] else v['detail']}")
    print(f"total {status['total_minutes']:.0f} min; archives in {out}:")
    for f in sorted(out.glob("*.zip")):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
