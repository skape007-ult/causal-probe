"""The notebook cells that decide where the output goes and check that it arrived.

Background (notes/log.md, 13 Nov 22:00): a notebook written for Colab was run on Kaggle, whose image also has a
/content folder; it wrote its archives to /content/out, which Kaggle does not keep. These tests run the cells'
own code, including that case, and check that the final cell reports a missing archive in STATUS.txt without raising,
and that the ingest refuses a run whose STATUS.txt is missing or not OK.
"""
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_kaggle_notebook  # noqa: E402
import make_round2_notebooks as gen  # noqa: E402
from notebook_cells import FINAL_CHECK, PLATFORM_SETUP  # noqa: E402


def functions(cell: str, *names: str) -> dict:
    """Execute only the function definitions of a cell (not its top-level statements)."""
    import ast
    tree = ast.parse(cell)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import)) and getattr(n, "name", "import") in (*names, "import")]
    ns = {"os": os}
    exec(compile(tree, "cell", "exec"), ns)
    return ns


def test_platform_choice_kaggle_wins_over_content_folder():
    f = functions(PLATFORM_SETUP, "choose_dirs", "under")
    choose, under = f["choose_dirs"], f["under"]
    # the case that lost a run: Kaggle's image has both /kaggle and /content
    plat, work, temp, keep = choose({}, lambda p: p in ("/kaggle", "/content"))
    assert (plat, work, keep) == ("kaggle", "/kaggle/working/kaggle_out", "/kaggle/working") and under(work, keep)
    assert not temp.startswith("/kaggle/working")           # scratch files must not fill the kept directory
    # Kaggle recognised by its environment variable alone
    assert choose({"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, lambda p: p == "/content")[:2] == ("kaggle", "/kaggle/working/kaggle_out")
    # Colab: /content without /kaggle
    assert choose({}, lambda p: p == "/content")[:2] == ("colab", "/content/out")
    assert choose({"COLAB_RELEASE_TAG": "x"}, lambda p: False)[0] == "colab"
    # neither: a local folder, no rule about where it must be
    plat, work, _, keep = choose({}, lambda p: False)
    assert plat == "local" and keep is None and work.endswith("kaggle_out")
    # the old behaviour would have produced this pair, and the check must reject it
    assert not under("/content/out", "/kaggle/working")
    assert not under("/kaggle/working_other/x", "/kaggle/working")


def _zip(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("results/x.json", json.dumps({"a": list(range(2000))}))


def test_check_outputs_can_fail(tmp_path, capsys):
    f = functions(PLATFORM_SETUP, "under")
    ns = functions(FINAL_CHECK, "check_outputs"); ns["under"] = f["under"]
    check = ns["check_outputs"]
    keep = tmp_path / "kaggle" / "working"
    good = keep / "kaggle_out"
    _zip(good / "results_r.zip"); _zip(good / "cache_r_m.zip")
    exp = ["results_r.zip", "cache_r_m.zip"]
    assert check(str(good), str(keep), exp) == []
    assert "results_r.zip" in capsys.readouterr().out                      # the listing names the files
    # the archives exist but in a directory the platform does not keep (what happened on 13 Nov)
    lost = tmp_path / "content" / "out"
    _zip(lost / "results_r.zip"); _zip(lost / "cache_r_m.zip")
    assert any("not under" in q for q in check(str(lost), str(keep), exp))
    # one archive missing; one empty; nothing expected at all
    (good / "cache_r_m.zip").unlink()
    assert any("missing" in q and "cache_r_m.zip" in q for q in check(str(good), str(keep), exp))
    (good / "cache_r_m.zip").write_bytes(b"")
    assert any("empty or not a zip" in q for q in check(str(good), str(keep), exp))
    assert check(str(good), str(keep), []) != []


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    saved = dict(os.environ)
    monkeypatch.chdir(tmp_path)
    for k in ("HF_TOKEN", "KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG", "COLAB_GPU"):
        monkeypatch.delenv(k, raising=False)
    yield tmp_path
    os.environ.clear(); os.environ.update(saved)


def _cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.skipif(os.path.isdir("/kaggle") or os.path.isdir("/content"), reason="written for a machine that is neither Kaggle nor Colab")
def test_round2_notebook_cells_end_to_end(clean_env, capsys):
    """Settings, setup, summary, download and final-check cells of a generated notebook, with the runner's output faked."""
    r = next(x for x in gen.RUNS if x["file"] == "kaggle_r2_qwen1.5b.ipynb")
    settings, setup, _clone, _env, _run, summary, final = _cells(gen.notebook(r, "0" * 40))
    ns = {}
    exec(settings, ns); exec(setup, ns)
    import types
    ns["p"] = types.SimpleNamespace(returncode=0)                       # the run cell's process: ended normally
    assert ns["PLATFORM"] == "local" and Path(ns["WORK"]) == clean_env / "kaggle_out"
    # what the runner leaves behind
    ns["REPO_DIR"] = str(clean_env / "repo"); (clean_env / "repo" / "results").mkdir(parents=True)
    run = ns["RUN_NAME"]
    status = {"planned": ["r2.qwen1.5b.cache"], "stages": {"r2.qwen1.5b.cache": {"ok": True, "minutes": 9.0, "detail": "ok"}},
              "models_run": ["qwen1.5b"], "export": {"zip_megabytes": {f"cache_{run}_qwen1.5b.zip": 434.8}}, "finished": True}
    (clean_env / "repo" / "results" / f"round2_status_{run}.json").write_text(json.dumps(status))
    for z in (f"results_{run}.zip", f"cache_{run}_qwen1.5b.zip"):
        _zip(Path(ns["WORK"]) / z)
    exec(summary, ns); exec(final, ns)
    work = Path(ns["WORK"])
    assert ns["expected"] == [f"results_{run}.zip", f"cache_{run}_qwen1.5b.zip"] and ns["problems"] == []
    assert "OUTPUT CHECK PASSED" in capsys.readouterr().out
    st = (work / "STATUS.txt").read_text().splitlines()
    assert st[0] == "OK" and f"run: {run}" in st and "commit: " + "0" * 40 in st and "problems: none" in st
    member = f"results/round2_output_status_{run}.txt"
    with zipfile.ZipFile(work / f"results_{run}.zip") as z:               # the copy that travels inside the results archive
        assert z.read(member).decode().splitlines()[0] == "OK" and "results/x.json" in z.namelist()
    # the activation archive is not there: the last cell must NOT raise; it writes FAILED and says so loudly
    (work / f"cache_{run}_qwen1.5b.zip").unlink()
    exec(summary, ns); exec(final, ns)
    out = capsys.readouterr().out
    assert "OUTPUT CHECK FAILED" in out and "!" * 100 in out and ns["OUTPUT_STATE"] == "FAILED"
    st = (work / "STATUS.txt").read_text()
    assert st.splitlines()[0] == "FAILED" and "expected archive is missing" in st and "MISSING" in st
    with zipfile.ZipFile(work / f"results_{run}.zip") as z:               # the copy was replaced, not duplicated
        assert z.namelist().count(member) == 1 and z.read(member).decode().splitlines()[0] == "FAILED"
    # the runner never reached its export step: also FAILED, even with the small archive present
    _zip(work / f"cache_{run}_qwen1.5b.zip")
    status.pop("finished")
    (clean_env / "repo" / "results" / f"round2_status_{run}.json").write_text(json.dumps(status))
    exec(summary, ns); exec(final, ns)
    assert ns["OUTPUT_STATE"] == "FAILED" and "did not reach its export step" in (work / "STATUS.txt").read_text()
    # a run that skipped itself (no access to the model, too little GPU memory): SKIPPED, with the reason
    status.update(finished=True, models_run=[], skipped={"gemma9b": "needs 2 GPUs, found 1"}, export={"zip_megabytes": {}})
    status["stages"] = {}
    (clean_env / "repo" / "results" / f"round2_status_{run}.json").write_text(json.dumps(status))
    exec(summary, ns); exec(final, ns)
    st = (work / "STATUS.txt").read_text()
    assert st.splitlines()[0] == "SKIPPED" and "needs 2 GPUs, found 1" in st and "RUN SKIPPED" in capsys.readouterr().out
    # the runner ended with an error while an earlier attempt's finished status and archives are still there: not OK (audit 7, F10)
    status.update(models_run=["qwen1.5b"], skipped={}, export={"zip_megabytes": {f"cache_{run}_qwen1.5b.zip": 434.8}})
    status["stages"] = {"r2.qwen1.5b.cache": {"ok": True, "minutes": 9.0, "detail": "ok"}}
    (clean_env / "repo" / "results" / f"round2_status_{run}.json").write_text(json.dumps(status))
    exec(summary, ns); exec(final, ns)
    assert ns["OUTPUT_STATE"] == "OK"
    ns["p"] = types.SimpleNamespace(returncode=2)
    exec(summary, ns); exec(final, ns)
    assert ns["OUTPUT_STATE"] == "FAILED" and "exit code 2" in (work / "STATUS.txt").read_text()
    del ns["p"]
    exec(summary, ns); exec(final, ns)
    assert ns["OUTPUT_STATE"] == "FAILED" and "not executed" in (work / "STATUS.txt").read_text()
    ns["p"] = types.SimpleNamespace(returncode=0)
    # a half-written status file must not stop the summary cell (then the last cell would never run)
    (clean_env / "repo" / "results" / f"round2_status_{run}.json").write_text('{"planned": ["x"], "stag')
    exec(summary, ns); exec(final, ns)
    assert ns["OUTPUT_STATE"] == "FAILED" and "could not be read" in (work / "STATUS.txt").read_text()
    # and the check survives its own failure (an unwritable output directory) without raising
    ns["WORK"] = str(clean_env / "does" / "not" / "exist")
    exec(final, ns)
    assert ns["OUTPUT_STATE"] == "FAILED"


@pytest.mark.skipif(os.path.isdir("/kaggle") or os.path.isdir("/content"), reason="written for a machine that is neither Kaggle nor Colab")
def test_gated_model_without_token_stops_in_the_first_cell(clean_env):
    r = next(x for x in gen.RUNS if x["file"] == "kaggle_r2_gemma.ipynb")
    settings, setup = _cells(gen.notebook(r, "0" * 40))[:2]
    ns = {}
    exec(settings, ns)
    assert ns["GATED_REPOS"] == "google/gemma-2-2b-it"
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        exec(setup, ns)
    os.environ["HF_TOKEN"] = "hf_test_value_not_a_real_token"
    ns["GATED_REPOS"] = ""                                             # no network in the test suite: the access check is tested below
    exec(setup, ns)                                                    # with a token in the environment it goes through
    assert "_tok" not in ns


def test_access_check_and_gpu_count(clean_env, monkeypatch):
    """The first cell's two other early stops: no access to gated weights, and too few GPUs."""
    import urllib.error
    import urllib.request
    f = functions(PLATFORM_SETUP, "check_access")
    seen = {}

    class FakeOpener:
        def __init__(self, code):
            self.code = code

        def open(self, req, timeout=None):
            seen["url"], seen["auth"], seen["method"] = req.full_url, req.get_header("Authorization"), req.get_method()
            if self.code != 200:
                raise urllib.error.HTTPError(req.full_url, self.code, "x", {}, None)

    for code, ok in ((200, True), (302, True), (307, True), (401, False), (403, False), (404, False), (503, None)):
        monkeypatch.setattr(urllib.request, "build_opener", lambda *a, code=code: FakeOpener(code))
        got, why = f["check_access"]("google/gemma-2-9b-it", "tok")
        assert got is ok and (ok or why), (code, got, why)

    class Down:
        def open(self, req, timeout=None):
            raise urllib.error.URLError("no route")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Down())
    assert f["check_access"]("google/gemma-2-9b-it", "tok")[0] is None       # not asked is not the same as refused
    assert seen == {"url": "https://huggingface.co/google/gemma-2-9b-it/resolve/main/config.json", "auth": "Bearer tok", "method": "HEAD"}
    # in the generated Gemma-2-9B notebook: a rejected token stops the first cell; so does a single GPU on Kaggle
    r = next(x for x in gen.RUNS if x["file"] == "kaggle_r3_gemma9b.ipynb")
    settings, setup = _cells(gen.notebook(r, "0" * 40))[:2]
    ns = {}
    exec(settings, ns)
    assert ns["MIN_GPUS"] == 2 and ns["GATED_REPOS"] == "google/gemma-2-9b-it" and "preflight" in ns["JOBS"]
    os.environ["HF_TOKEN"] = "hf_test_value_not_a_real_token"
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: FakeOpener(401))
    with pytest.raises(RuntimeError, match="No access to google/gemma-2-9b-it"):
        exec(setup, ns)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: FakeOpener(200))
    exec(setup, ns)                                                    # access confirmed; on a laptop the GPU count is not enforced
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Down())
    exec(setup, ns)                                                    # Hugging Face unreachable: a warning, not a stop
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: FakeOpener(200))
    one_gpu = setup.replace('PLATFORM != "local" and globals()', "globals()").replace("_n_gpu = len(", "_n_gpu = 1 + 0 * len(")
    assert one_gpu != setup
    with pytest.raises(RuntimeError, match="needs 2 GPU"):
        exec(one_gpu, ns)


def test_every_notebook_ends_with_the_output_check():
    nbs = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
    assert len(nbs) >= 9
    for p in nbs:
        cells = _cells(json.loads(p.read_text()))
        assert "OUTPUT CHECK FAILED" in cells[-1] and "STATUS.txt" in cells[-1] and "raise " not in cells[-1], p.name
        assert "choose_dirs(os.environ, os.path.isdir)" in cells[1], p.name
        assert '"/content/out" if' not in "".join(cells) and 'WORK = "/kaggle/working" if' not in "".join(cells), p.name
        # the pinned commit is a commit of this repository (a placeholder pin would fail in the clone cell)
        import re
        import subprocess
        pin = re.search(r'COMMIT\s*=\s*"([^"]*)"', cells[0]).group(1)
        assert re.fullmatch(r"[0-9a-f]{40}", pin), (p.name, pin)
        assert subprocess.run(["git", "cat-file", "-e", pin + "^{commit}"], cwd=p.parent.parent).returncode == 0, (p.name, pin)
        # on Colab the download happens in the last cell, after the status exists (audit 7, F2)
        assert cells[-1].index("write_status(WORK") < cells[-1].index("files.download"), p.name
        for i, c in enumerate(cells):
            compile(c, f"{p.name}:cell{i}", "exec")
    assert callable(make_kaggle_notebook.main)


# ---------------------------------------------------------------- the ingest and STATUS.txt
def _fake_run(d: Path, run: str, commit: str, status_in_zip: str | None = None):
    d.mkdir(parents=True)
    with zipfile.ZipFile(d / f"results_{run}.zip", "w") as z:
        z.writestr(f"results/round2_environment_{run}.json", json.dumps({"git_commit": commit}))
        z.writestr(f"results/round2_status_{run}.json", json.dumps(
            {"run": run, "jobs": ["export"], "planned": ["data"], "stages": {"data": {"ok": True, "minutes": 0.1, "detail": "ok"}},
             "models_run": [], "export": {"zip_megabytes": {}}, "finished": True, "total_minutes": 1}))
        if status_in_zip is not None:
            z.writestr(f"results/round2_output_status_{run}.txt", status_in_zip)


_REAL_GIT_BLOB = {}


def _ingest(monkeypatch, d: Path, run: str, writes_status, *extra):
    import subprocess

    import kaggle_ingest
    real = _REAL_GIT_BLOB.setdefault("f", kaggle_ingest.git_blob)
    if writes_status is None:                    # the real rule: what scripts/notebook_cells.py says at the run's commit
        monkeypatch.setattr(kaggle_ingest, "git_blob", real)
    else:
        monkeypatch.setattr(kaggle_ingest, "git_blob", lambda c, path: (b"STATUS.txt" if writes_status else b"old cells")
                            if path == "scripts/notebook_cells.py" else real(c, path))
    monkeypatch.setattr(sys, "argv", ["kaggle_ingest.py", "--run", run, "--dir", str(d), "--dry-run", *extra])
    with pytest.raises(SystemExit) as e:
        kaggle_ingest.main()
    return e.value.code


def test_ingest_refuses_missing_or_bad_status(tmp_path, monkeypatch, capsys):
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    run = "r9_statustest"
    ok = f"OK\nrun: {run}\ncommit: {head}\nproblems: none\n"
    # OK beside the archives: validation passes (dry run exits 0)
    _fake_run(tmp_path / "a", run, head); (tmp_path / "a" / "STATUS.txt").write_text(ok)
    assert _ingest(monkeypatch, tmp_path / "a", run, True) == 0
    # OK only as the copy inside the results archive: passes
    _fake_run(tmp_path / "b", run, head, status_in_zip=ok)
    assert _ingest(monkeypatch, tmp_path / "b", run, True) == 0
    # missing, from a notebook that writes it: refused
    _fake_run(tmp_path / "c", run, head)
    assert "STATUS.txt is missing" in str(_ingest(monkeypatch, tmp_path / "c", run, True))
    # missing, from a notebook that predates it: accepted with a note
    assert _ingest(monkeypatch, tmp_path / "c", run, False) == 0 and "did not write STATUS.txt" in capsys.readouterr().out
    # FAILED, SKIPPED, empty, or another run's file: refused
    for i, txt in enumerate([f"FAILED\nrun: {run}\nproblems:\n  - expected archive is missing: x.zip\n",
                             f"SKIPPED\nrun: {run}\nproblems:\n  - the run skipped itself: needs 2 GPUs\n", "",
                             f"OK\nrun: r2_other\n"]):
        _fake_run(tmp_path / f"d{i}", run, head); (tmp_path / f"d{i}" / "STATUS.txt").write_text(txt)
        msg = str(_ingest(monkeypatch, tmp_path / f"d{i}", run, True))
        assert msg.startswith("ERROR") and ("not OK" in msg or "another run" in msg), msg
    # --allow-partial does not override it; --ignore-status does, and says so
    assert str(_ingest(monkeypatch, tmp_path / "d0", run, True, "--allow-partial")).startswith("ERROR")
    assert _ingest(monkeypatch, tmp_path / "d0", run, True, "--ignore-status") == 0 and "OVERRIDDEN" in capsys.readouterr().out


def test_stale_status_beside_the_archives_and_unknown_commit(tmp_path, monkeypatch, capsys):
    """Seventh audit, F4 and F5."""
    import subprocess
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    run = "r9_statustest"
    ok = f"OK\nrun: {run}\ncommit: {head}\nwritten: 2026-11-14T01:00:00Z\nproblems: none\n"
    # loose OK from an earlier download, copy inside the archive FAILED: refused
    _fake_run(tmp_path / "a", run, head, status_in_zip=f"FAILED\nrun: {run}\ncommit: {head}\nwritten: 2026-11-14T05:00:00Z\nproblems:\n  - x\n")
    (tmp_path / "a" / "STATUS.txt").write_text(ok)
    assert "do not agree" in str(_ingest(monkeypatch, tmp_path / "a", run, True))
    # both OK but written at different times: also refused; identical: accepted; a byte-order mark does not matter
    _fake_run(tmp_path / "b", run, head, status_in_zip=ok.replace("01:00:00", "02:00:00")); (tmp_path / "b" / "STATUS.txt").write_text(ok)
    assert "do not agree" in str(_ingest(monkeypatch, tmp_path / "b", run, True))
    _fake_run(tmp_path / "c", run, head, status_in_zip=ok); (tmp_path / "c" / "STATUS.txt").write_text("\ufeff" + ok.replace("\n", "\r\n"))
    assert _ingest(monkeypatch, tmp_path / "c", run, True) == 0
    # OK written for another commit than the run's environment record: refused
    _fake_run(tmp_path / "d", run, head); (tmp_path / "d" / "STATUS.txt").write_text(ok.replace(head, "f" * 40))
    assert "was written for commit" in str(_ingest(monkeypatch, tmp_path / "d", run, True))
    # no usable environment record and no status: the check does not switch itself off, with or without --allow-partial
    for i, env_commit in enumerate(["", "not-a-commit", "e" * 40]):
        _fake_run(tmp_path / f"e{i}", run, env_commit)
        assert "STATUS.txt is missing" in str(_ingest(monkeypatch, tmp_path / f"e{i}", run, None, "--allow-partial"))
    # the real rule on real commits: 6d9ce8d has the shared cells but no STATUS.txt (accepted without); 041f11e writes it (refused without)
    _fake_run(tmp_path / "f", run, "6d9ce8dbe7146b10c10e7285c276a786d3727fd2")
    assert _ingest(monkeypatch, tmp_path / "f", run, None) == 0
    _fake_run(tmp_path / "g", run, "041f11e" + subprocess.check_output(["git", "rev-parse", "041f11e"], cwd=Path(__file__).resolve().parents[1], text=True).strip()[7:])
    assert "STATUS.txt is missing" in str(_ingest(monkeypatch, tmp_path / "g", run, None))


def test_two_gpu_plan_fits_real_t4s():
    """Seventh audit, F1: a Kaggle T4 reports 15.6e9 bytes = 14.5 GiB, and the plan for Gemma-2-9B must pass on two of them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("preflight_model", Path(__file__).resolve().parents[1] / "scripts" / "preflight_model.py")
    pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
    from causal_probe import models
    from causal_probe.config import PARAMS_B
    caps = models.multi_gpu_caps_gib(2)
    t4 = 15.6e9 / 2 ** 30
    for name in ("gemma9b", "qwen7b"):
        w = PARAMS_B[name] * 1e9 * 2 / 2 ** 30
        ok, why = pm.fits(w, [t4, t4], caps)
        assert ok, (name, why)
        assert pm.fits(w, [14.4, 14.4], caps)[0], name                 # a little less than the recorded size still passes
    w9 = PARAMS_B["gemma9b"] * 1e9 * 2 / 2 ** 30
    assert not pm.fits(w9, [t4], caps)[0]                                # one T4
    assert not pm.fits(w9, [t4, 11.0], caps)[0]                          # a second card that is too small
    assert not pm.fits(30.0, [t4, t4], caps)[0]                          # a model that is too large
    assert 17.0 < w9 < 17.4 and sum(caps) >= w9 + 4                      # 17.2 GiB of weights, at least 4 GiB of slack in the plan
