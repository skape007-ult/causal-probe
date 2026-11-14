"""Round-2 GPU jobs (free Kaggle or Colab GPU; see KAGGLE.md, "Round 2"). Also runs anywhere else.

Same machinery as scripts/kaggle_run.py: every stage is one subprocess, is skipped if its marker
file exists (re-running resumes), has a time limit, and may fail without stopping the stages after
it. The status of every stage goes to results/round2_status_<run>.json; the small results archive is
refreshed after every stage.

  python scripts/gpu_round2.py --run r2_qwen3b --models qwen3b \
      --jobs cache,controls,controls_hyp,patch,patch_same,feature,export --out /kaggle/working
  python scripts/gpu_round2.py --run test --models qwen0.5b --jobs ... --tiny --out <dir>     # local smoke test

Jobs (each applies to every model in --models, in this order):
  preflight     can this model run here at all? (scripts/preflight_model.py: access to the weights, GPU memory,
                disk; then a real load and 16 items through the cache code.) If not, the model is skipped with
                the reason in the status file, the remaining jobs are not attempted for it, and the notebook's
                last cell writes SKIPPED into STATUS.txt. Used for Gemma-2-9B on two T4s.
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
  adjacency     interventions on the encoding of adjacency (steps, translation by the class-mean
                difference, LEACE erasure) with controls from labels shuffled within hypothesis
                string, at four sites (final token / end of hypothesis x best layer / first layer
                at 50% of the best layer's excess AUROC): scripts/17_adjacency_test.py
  feature       the first version of that test (scripts/16_feature_causal.py), superseded after the
                code review; kept so that its results can be reproduced
  settle        one steering arm that ranked first among 20 controls at the end of the hypothesis on the debiased pool:
                independent draws of the direction (other halves of the blocks), controls that keep adjacency, the
                adjacency direction itself, neighbouring layers, smaller steps (scripts/07d_steering_settle.py;
                --settle-arm names the direction of the primary test)
  Options that change the size of the intervention jobs (round 3; the output names get a suffix, so that these
  runs never replace the round-2 results):
    --n-shuf 100       100 shuffled-label control directions instead of 20 in controls, controls_hyp, adjacency
                       (files ..._c100). The sites are those of the 20-control run (the selection files are shared).
    --sites best       adjacency at the two best-layer sites only
    --control-tags c2c_debiased   steering controls on the debiased pool only
    --n-pairs 240      patch and patch_same on up to 240 pairs per kind instead of 60 (files ..._n240), activation
                       patching only, no random-donor control. The pool holds about 210 label-changing and 230
                       label-preserving pairs per kind at most.
    --no-resid         export per-item logits and raw intervention outputs only, not the activations
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
JOBS = ["preflight", "behaviour", "cache", "settle", "controls", "controls_hyp", "patch", "patch_same", "adjacency", "feature", "export"]
SITES = [("final", "best"), ("content_end", "best"), ("final", "first90"), ("content_end", "first90")]      # first version (feature)
SITES2 = [("final", "best"), ("content_end", "best"), ("final", "first50"), ("content_end", "first50")]    # adjacency


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
    ap.add_argument("--n-shuf", type=int, default=20, help="shuffled-label control directions (controls, controls_hyp, adjacency)")
    ap.add_argument("--sites", default="all", choices=["all", "best"], help="adjacency sites: all four, or the two best-layer sites")
    ap.add_argument("--control-tags", default="c2c_debiased,c2c_probe", help="item sets for the steering controls")
    ap.add_argument("--n-pairs", type=int, default=60, help="patching pairs per kind (patch, patch_same)")
    ap.add_argument("--settle-arm", default="gold_diffmeans", choices=["gold_diffmeans", "gold_probe"], help="direction of the primary test of the settle job")
    ap.add_argument("--no-resid", action="store_true", help="export no activations (logits and raw intervention outputs only)")
    a = ap.parse_args()
    jobs = [j for j in a.jobs.split(",") if j]
    assert all(j in JOBS for j in jobs), jobs
    if a.tiny:                      # a smoke test never shares markers, status, result or selection files with a real run
        a.run = a.run + "_tiny"
        os.environ["CAUSAL_PROBE_RESULTS"] = str(Path(os.environ.get("CAUSAL_PROBE_RESULTS", ROOT / "results")) / "_tiny")
        # ... nor the activation cache: a 64-item cache left by a smoke test would be found "already complete" by the real
        # run in the same session, and its numbers would carry the real run's name (seventh audit, F6)
        os.environ["CAUSAL_PROBE_CACHE"] = str(Path(os.environ.get("CAUSAL_PROBE_CACHE", ROOT / "cache")) / "_tiny")
    fallback = dict(p.split(":") for p in a.fallback.split(",") if p)

    sys.path.insert(0, str(ROOT / "src"))
    from causal_probe import cache as C
    from causal_probe.config import CACHE, RESULTS

    out = Path(a.out)
    # Kaggle keeps only /kaggle/working after a commit run (a run on 13 Nov wrote to /content/out there and lost its output)
    if (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle")) and not (os.path.abspath(out) + "/").startswith("/kaggle/working/"):
        sys.exit(f"refusing to start: on Kaggle only /kaggle/working is kept, and --out is {out}")
    out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    logs = RESULTS / f"round2_logs_{a.run}"; logs.mkdir(exist_ok=True)
    marks = CACHE / f"_round2_markers_{a.run}"; marks.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS / f"round2_status_{a.run}.json"

    def save_status():        # written through a temporary file: a half-written status would stop the notebook's summary cell
        tmp = status_path.with_name(status_path.name + ".tmp")
        tmp.write_text(json.dumps(status, indent=1))
        os.replace(tmp, status_path)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    for k in ("export", "all_ok", "finished"):      # only this invocation may declare the run finished
        status.pop(k, None)
    status.pop("skipped", None)
    status.update(run=a.run, models=a.models.split(","), jobs=jobs, tiny=a.tiny, planned=[],
                  options={"n_shuf": a.n_shuf, "sites": a.sites, "control_tags": a.control_tags, "n_pairs": a.n_pairs, "no_resid": a.no_resid})
    save_status()
    skip = [s for s in a.skip.split(",") if s]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               TOKENIZERS_PARALLELISM="false")

    def zip_small():
        with zipfile.ZipFile(out / f"results_{a.run}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(RESULTS.rglob("*")):
                if f.is_file() and f.stat().st_size < 20e6:
                    z.write(f, Path("results") / f.relative_to(RESULTS))

    def stage(name: str, *cmd: str, minutes: float | None = None, optional: bool = False, env_extra: dict | None = None) -> bool:
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
                p = subprocess.run([PY, *cmd], cwd=ROOT, env={**env, **(env_extra or {})}, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=(minutes or a.stage_minutes) * 60)
            ok, why = p.returncode == 0, f"exit code {p.returncode}"
        except subprocess.TimeoutExpired:
            ok, why = False, f"timeout after {(minutes or a.stage_minutes):.0f} min"
        mins = (time.time() - t) / 60
        tail = log.read_text(errors="replace").strip().splitlines()[-8:]
        status["stages"][name] = {"ok": ok, "minutes": round(mins, 1), "detail": "ok" if ok else why, "log_tail": tail}
        status["total_minutes"] = round((time.time() - T0) / 60, 1)
        save_status()
        print(f"[{'ok' if ok else 'FAILED: ' + why}] {name}  {mins:.1f} min", flush=True)
        for line in tail:
            print("        | " + line[:200], flush=True)
        if ok:
            (marks / name).write_text("done")
        zip_small()
        return ok

    stage("data", "scripts/download_data.py", minutes=30)
    # the environment record goes to a per-run file; results/kaggle_environment.json stays the first Kaggle run's record
    stage("environment", "scripts/kaggle_env.py", "--out", f"round2_environment_{a.run}.json", minutes=10)

    # the smoke test builds the two caches in full (the selection/report split needs whole blocks);
    # only the number of items each intervention stage runs is reduced
    models_run = []
    cs = "" if a.n_shuf == 20 else f"_c{a.n_shuf}"            # suffix of runs with another number of control directions
    ps = "" if a.n_pairs == 60 else f"_n{a.n_pairs}"          # suffix of runs with another number of patching pairs
    for M0 in a.models.split(","):
        M = M0
        # ------------------------------------------------------------ preflight: can this model run here?
        if "preflight" in jobs:
            t_pre = time.time() - 1
            ok = stage(f"r2.{M}.preflight.resources", "scripts/preflight_model.py", "--model", M, "--check", "resources", minutes=6)
            # the load check writes 16 items to a scratch cache, never to the cache the real stages use
            ok = ok and stage(f"r2.{M}.preflight.load", "scripts/preflight_model.py", "--model", M, "--check", "load", minutes=50,
                              env_extra={"CAUSAL_PROBE_CACHE": str(CACHE / "_preflight")})
            if not ok:
                pf = RESULTS / f"preflight_{M}.json"
                try:                                   # the file may be missing, half-written, or left by an earlier attempt
                    rec_ = json.loads(pf.read_text()) if pf.exists() and pf.stat().st_mtime >= t_pre else {}
                except Exception:
                    rec_ = {}
                why = rec_.get("verdict")
                bad = [n_ for n_ in (f"r2.{M}.preflight.resources", f"r2.{M}.preflight.load") if not status["stages"].get(n_, {"ok": True})["ok"]]
                if not why or rec_.get("ok"):          # the check crashed, timed out or was skipped before writing a verdict of its own
                    why = (f"preflight stage {bad[0]} did not finish ({status['stages'][bad[0]].get('detail', '?')}); see its log" if bad
                           else "the preflight was not run (skipped with SKIP_STAGES, or stopped): without it this model is not attempted")
                status.setdefault("skipped", {})[M] = why
                save_status()
                print("\n" + "!" * 100 + f"\nSKIPPING {M}: {why}\n" + "!" * 100, flush=True)
                continue
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
            save_status()
            # report whatever was cached, also after a timeout (the debiased pool comes first)
            stage(f"r2.{M}.behaviour.report", "scripts/15_answer_bias.py", minutes=10)
        # ------------------------------------------------------------ cache for the interventions
        have_cache = True
        if "cache" in jobs:
            have_cache = stage(f"r2.{M}.cache", "scripts/01_build_cache.py", "--model", M, "--tags", "c2c_probe,c2c_debiased",
                               "--no-gen", "--max-minutes", "50", minutes=58)
            if not have_cache and M in fallback and status["stages"].get(f"r2.{M}.cache", {}).get("minutes", 99) < 12:
                status["planned"].remove(f"r2.{M}.cache")        # replaced by the fallback model's stage
                M = fallback[M]
                have_cache = stage(f"r2.{M}.cache", "scripts/01_build_cache.py", "--model", M, "--tags", "c2c_probe,c2c_debiased",
                                   "--no-gen", "--max-minutes", "50", minutes=58)
        models_run.append(M)
        n_c = ["--n", "10"] if a.tiny else ["--n", "150"]
        stride = ["--select-stride", "6" if a.tiny else "2"]
        if have_cache:
            if "settle" in jobs:
                stage(f"r2.{M}.settle", "scripts/07d_steering_settle.py", "--model", M, "--arm", a.settle_arm, "--sub-batch", "16",
                      *(["--n", "10", "--n-shuf", "6", "--n-other", "4", "--n-splits", "3", "--max-minutes", "20", "--allow-new-site"] if a.tiny
                        else ["--max-minutes", "75"]), minutes=88)
            ctags = [t_ for t_ in a.control_tags.split(",") if t_]
            # 07c runs 1 + 2 x (3 + 100 isotropic + 2 x n_shuf) vectors per item: the time limits grow with that number
            kc = (103 + 2 * a.n_shuf) / 143
            new_site = ["--allow-new-site"] if a.tiny else []          # a smoke test has no 20-control result to agree with
            c_extra = ["--n-shuf", str(a.n_shuf), "--out-suffix", cs, *new_site] if cs else []
            if "controls" in jobs:
                for tag in ctags:
                    stage(f"r2.{M}.controls{cs}.{tag}", "scripts/07c_steering_controls.py", "--model", M, "--tag", tag,
                          "--sub-batch", "16", "--max-minutes", f"{30 * kc:.0f}", *stride, *n_c, *c_extra, minutes=30 * kc + 10)
            if "controls_hyp" in jobs:
                for tag in ctags:
                    stage(f"r2.{M}.controls_hyp{cs}.{tag}", "scripts/07c_steering_controls.py", "--model", M, "--tag", tag,
                          "--position", "content_end", "--sub-batch", "16", "--max-minutes", f"{50 * kc:.0f}", *stride, *n_c, *c_extra,
                          minutes=50 * kc + 8)
            if "adjacency" in jobs:
                # with more controls each item costs more; the script shortens itself to a template-balanced prefix that
                # fits --max-minutes and records how many items it ran
                a_min = 55 if not cs else 110
                for pos, spec in (SITES2 if a.sites == "all" else SITES2[:2]):
                    stage(f"r2.{M}.adjacency{cs}.{pos}.{spec}", "scripts/17_adjacency_test.py", "--model", M, "--position", pos,
                          "--layer-spec", spec, "--sub-batch", "16", "--max-minutes", str(a_min),
                          *(["--per-template", "24", "--n-shuf", "3", "--select-stride", "6"] if a.tiny
                            else ["--per-template", "100", "--select-stride", "1", "--n-shuf", str(a.n_shuf)]),
                          *(["--suffix", cs, *new_site] if cs else []), minutes=a_min + 12)
            if "feature" in jobs:
                for pos, spec in SITES:
                    stage(f"r2.{M}.feature.{pos}.{spec}", "scripts/16_feature_causal.py", "--model", M, "--position", pos,
                          "--layer-spec", spec, "--sub-batch", "16", "--max-minutes", "45",
                          *(["--per-template", "4", "--n-shuf", "3", "--n-iso", "2", "--select-stride", "6"] if a.tiny
                            else ["--per-template", "100", "--select-stride", "1"]), minutes=70)   # fitting ~130 probes on CPU comes first
        pp = ["--n-pairs", "6", "--layer-stride", "6", "--batch", "4"] if a.tiny else ["--n-pairs", str(a.n_pairs), "--layer-stride", "2", "--batch", "8"]
        p_min = 40 if not ps else min(175, 40 * a.n_pairs / 60 + 15)     # no checkpoint inside 08_patching.py: the limit must hold the whole stage
        if "patch_same" in jobs:
            stage(f"r2.{M}.patch_same{ps}", "scripts/08_patching.py", "--model", M, "--label-mode", "same", "--suffix", "_same" + ps,
                  "--skip-components", *pp, minutes=p_min)
        if "patch" in jobs and not ps:
            stage(f"r2.{M}.patch", "scripts/08_patching.py", "--model", M, *pp,
                  *(["--n-comp-pairs", "4"] if a.tiny else ["--n-comp-pairs", "20"]), minutes=50)
        elif "patch" in jobs:       # larger pair count: activation patching only, no random-donor control (both exist in the 60-pair run)
            stage(f"r2.{M}.patch{ps}", "scripts/08_patching.py", "--model", M, "--suffix", ps, "--skip-components", "--no-donor-control",
                  *pp, minutes=p_min)

    # ------------------------------------------------------------------ export
    sizes, zsizes = {}, {}
    exp = out / "export"
    pos = tuple(a.export_positions.split(","))
    for M in models_run:
        for tag in ("c2c_debiased", "c2c_probe", "c2c_test", "cladder"):
            d = C.cache_dir(M, tag)
            if not (d / "meta.json").exists():
                continue
            full = tag == "c2c_debiased" and "export" in jobs and not a.no_resid
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
    status["finished"] = True
    save_status()
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
