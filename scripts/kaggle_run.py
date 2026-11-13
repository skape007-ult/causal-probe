"""Heavy GPU work, meant for a free Kaggle GPU session (see KAGGLE.md). Also runs anywhere else.

Runs a fixed list of stages, one subprocess each. Every stage
  * is skipped if its marker file exists (so re-running the cell resumes),
  * has a time limit, and
  * may fail without stopping the stages after it (the failure is recorded).
The status of every stage is written to results/kaggle_run_status.json and the small results
archive is refreshed after every stage, so a partial run still leaves something to download.

  python scripts/kaggle_run.py --gemma --large qwen3b --out /kaggle/working
  python scripts/kaggle_run.py --gemma --large qwen0.5b --tiny --fake-gemma qwen0.5b --out <dir>   # local smoke test

What it does
  Gemma (gemma-2-2b-it):  cache of the probe sets (+ debiased pool), steering at the layer chosen
      in results/confound_gemma2b_*.json (committed, from the laptop cache) and at two fixed
      depths, patching, SAE features + ablation (Gemma Scope via SAELens).
  Larger model:  full cache (all tags, generated answers, random-init control), behaviour, logit
      lens, steering. Probes and text baselines for it are NOT run here: the exported
      final-position cache is analysed on the laptop with the same scripts as the other models.
  Export:  results.zip (JSON, logs, per-item logits; a few MB) and one cache_<model>.zip per model
      with float16 residuals at the final token for the probe sets.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
T0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma", action="store_true")
    ap.add_argument("--large", default=None, help="model name from config (qwen3b, qwen7b-4bit); omit to skip")
    ap.add_argument("--out", default=str(ROOT / "kaggle_out"), help="where the zip files are written")
    ap.add_argument("--tiny", action="store_true", help="smoke test: a few dozen items per stage")
    ap.add_argument("--fake-gemma", default=None,
                    help="smoke test: run the 'Gemma' stages on this model with a random SAE dictionary")
    ap.add_argument("--skip", default="", help="comma-separated stage-name prefixes to skip")
    ap.add_argument("--export-positions", default="final", help="comma-separated: final,content_end,premise_end")
    ap.add_argument("--no-random-export", action="store_true", help="leave the random-init residuals out of the zips")
    ap.add_argument("--stage-minutes", type=float, default=90.0)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from causal_probe import cache as C                      # noqa: E402  (after sys.path)
    from causal_probe.config import CACHE, QUANT_4BIT, RESULTS

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    logs = RESULTS / "kaggle_logs"; logs.mkdir(exist_ok=True)
    marks = CACHE / "_kaggle_markers"; marks.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS / "kaggle_run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    skip = [s for s in a.skip.split(",") if s]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
               TOKENIZERS_PARALLELISM="false")

    def zip_small():
        """results/ (JSON, markdown, logs). Small; refreshed after every stage."""
        with zipfile.ZipFile(out / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(RESULTS.rglob("*")):
                if f.is_file() and f.stat().st_size < 20e6:
                    z.write(f, Path("results") / f.relative_to(RESULTS))

    def stage(name: str, *cmd: str, minutes: float | None = None) -> bool:
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
        tail = log.read_text(errors="replace").strip().splitlines()[-6:]
        status["stages"][name] = {"ok": ok, "minutes": round(mins, 1), "detail": "ok" if ok else why,
                                  "log_tail": tail}
        status["total_minutes"] = round((time.time() - T0) / 60, 1)
        status_path.write_text(json.dumps(status, indent=1))
        print(f"[{'ok' if ok else 'FAILED: ' + why}] {name}  {mins:.1f} min", flush=True)
        for line in tail:
            print("        | " + line[:200], flush=True)
        if ok:
            (marks / name).write_text("done")
        zip_small()
        return ok

    lim = ["--limit", "96"] if a.tiny else []
    n_steer = ["--n", "12"] if a.tiny else []
    stage("data", "scripts/download_data.py", minutes=30)
    stage("environment", "scripts/kaggle_env.py", minutes=10)

    # ------------------------------------------------------------------ Gemma
    G = a.fake_gemma or "gemma2b"
    if a.gemma:
        fake = ["--model", G, "--fake-sae"] if a.fake_gemma else []
        # cladder is always cached in full: the SAE stage needs all ten "nonsense" stories
        stage(f"gemma.{G}.cache_c2c", "scripts/01_build_cache.py", "--model", G, "--tags", "c2c_probe,c2c_debiased", "--no-gen", *lim)
        if stage(f"gemma.{G}.cache", "scripts/01_build_cache.py", "--model", G, "--tags", "c2c_probe,c2c_debiased,cladder", "--no-gen"):
            stage(f"gemma.{G}.cache_random", "scripts/01_build_cache.py", "--model", G, "--random", "--tags", "c2c_debiased", *lim)
            stage(f"gemma.{G}.cache_agreement", "scripts/kaggle_env.py", "--agreement", G, minutes=10)
            # steering: layer from 05_confound (committed JSON); 79 vectors per item in groups of 8
            for tag in ("c2c_probe", "cladder", "c2c_debiased"):
                stage(f"gemma.{G}.steer.{tag}", "scripts/07_steering.py", "--model", G, "--tag", tag, "--sub-batch", "8", *n_steer)
            for tag in ("c2c_probe", "cladder"):
                for d in ("0.5", "0.85"):                       # layer-choice sensitivity
                    stage(f"gemma.{G}.steer.{tag}.d{d}", "scripts/07_steering.py", "--model", G, "--tag", tag, "--sub-batch", "8",
                          "--rel-depth", d, "--suffix", f"_d{d}", *n_steer)

    # ------------------------------------------------------------------ larger model
    M = a.large
    if M:
        if a.tiny:   # smoke test: a prefix of each Corr2Cause set, but CLadder in full (a prefix of it has one label only)
            stage(f"large.{M}.cache_tiny", "scripts/01_build_cache.py", "--model", M, "--no-gen", *lim,
                  "--tags", "c2c_test,c2c_probe,c2c_debiased,c2c_test_renamed,c2c_test_sub_fewshot")
        ok = stage(f"large.{M}.cache", "scripts/01_build_cache.py", "--model", M, "--max-minutes", "80",
                   "--tags", "c2c_test,c2c_probe,cladder,c2c_debiased,c2c_test_renamed,c2c_test_sub_fewshot,cladder_sub_fewshot",
                   *(["--gen-limit", "40"] if a.tiny else []))
        if not ok and not a.tiny:   # the cache builder stops itself after 80 min and resumes on re-run
            ok = stage(f"large.{M}.cache_resume", "scripts/01_build_cache.py", "--model", M, "--max-minutes", "80",
                       "--tags", "c2c_test,c2c_probe,cladder,c2c_debiased,c2c_test_renamed,c2c_test_sub_fewshot,cladder_sub_fewshot")
        if ok:
            stage(f"large.{M}.behaviour", "scripts/02_behaviour.py", "--model", M, minutes=30)
            stage(f"large.{M}.lens", "scripts/03_logit_lens.py", "--model", M, minutes=30)
            for tag in ("c2c_probe", "cladder", "c2c_debiased"):
                stage(f"large.{M}.steer.{tag}", "scripts/07_steering.py", "--model", M, "--tag", tag, "--sub-batch", "8",
                      "--select-stride", "2", *n_steer)
            for d in ("0.5", "0.85"):
                stage(f"large.{M}.steer.c2c_probe.d{d}", "scripts/07_steering.py", "--model", M, "--tag", "c2c_probe", "--sub-batch", "8",
                      "--rel-depth", d, "--suffix", f"_d{d}", "--select-stride", "2", *n_steer)
            if M not in QUANT_4BIT:
                stage(f"large.{M}.cache_random", "scripts/01_build_cache.py", "--model", M, "--random",
                      "--tags", "c2c_debiased,c2c_probe,cladder", "--max-minutes", "80", *lim)

    # ------------------------------------------------------------------ Gemma, slower stages last
    if a.gemma and (marks / f"gemma.{G}.cache").exists():
        pp = ["--n-pairs", "6", "--n-comp-pairs", "4", "--layer-stride", "6", "--batch", "4"] if a.tiny else \
             ["--n-pairs", "40", "--n-comp-pairs", "20", "--layer-stride", "2", "--batch", "8"]
        stage(f"gemma.{G}.patching", "scripts/08_patching.py", "--model", G, *pp)
        if stage(f"gemma.{G}.sae_features", "scripts/11_sae.py", "--stage", "features", *fake, minutes=45):
            stage(f"gemma.{G}.sae_ablate", "scripts/11_sae.py", "--stage", "ablate", *fake,
                  *(["--n-per-struct", "4"] if a.tiny else []), minutes=60)

    # ------------------------------------------------------------------ export
    pos = tuple(a.export_positions.split(","))
    sizes = {}
    exp = out / "export"

    def export(model, tag, rand=False, resid=True):
        if not (C.cache_dir(model, tag, rand) / "meta.json").exists():
            return
        sizes[f"{model}{'-rand' if rand else ''}/{tag}"] = C.export_cache(model, tag, exp, rand, pos, with_resid=resid)

    if a.gemma:
        for tag in ("c2c_probe", "cladder", "c2c_debiased"):
            export(G, tag, resid=(tag == "c2c_debiased") or bool(a.fake_gemma))   # laptop already has the others
        if not a.no_random_export:
            export(G, "c2c_debiased", rand=True)
        for f in (CACHE / G).glob("patching.npz"):
            (exp / G).mkdir(parents=True, exist_ok=True)
            (exp / G / f.name).write_bytes(f.read_bytes())
    if M:
        for tag in ("c2c_probe", "cladder", "c2c_debiased"):
            export(M, tag)
            if not a.no_random_export:
                export(M, tag, rand=True)
        for tag in ("c2c_test", "c2c_test_renamed", "c2c_test_sub", "cladder_sub", "c2c_test_sub_fewshot", "cladder_sub_fewshot"):
            if (C.cache_dir(M, tag) / "meta.json").exists():
                export(M, tag, resid=False)          # per-item logits and generated answers only
            elif (C.cache_dir(M, tag) / "gen.json").exists():
                d = exp / M / tag; d.mkdir(parents=True, exist_ok=True)
                (d / "gen.json").write_bytes((C.cache_dir(M, tag) / "gen.json").read_bytes())
    zsizes = {}
    if exp.exists():
        for mdir in sorted(p for p in exp.iterdir() if p.is_dir()):
            zp = out / f"cache_{mdir.name}.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:      # float16 noise does not compress
                for f in sorted(mdir.rglob("*")):
                    if f.is_file():
                        z.write(f, Path("cache") / mdir.name / f.relative_to(mdir))
            zsizes[zp.name] = round(zp.stat().st_size / 1e6, 1)
        import shutil
        shutil.rmtree(exp)
    status["export"] = {"positions": list(pos), "bytes_per_cache_dir": sizes, "zip_megabytes": zsizes}
    status["total_minutes"] = round((time.time() - T0) / 60, 1)
    status["all_ok"] = all(s["ok"] for s in status["stages"].values())
    status_path.write_text(json.dumps(status, indent=1))
    zip_small()
    print("\n================ summary ================")
    for k, v in status["stages"].items():
        print(f"{'ok    ' if v['ok'] else 'FAILED'} {v['minutes']:6.1f} min  {k}  {'' if v['ok'] else v['detail']}")
    print(f"total {status['total_minutes']:.0f} min; archives in {out}:")
    for f in sorted(out.glob("*.zip")):
        print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
