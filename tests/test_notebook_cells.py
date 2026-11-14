"""The notebook cells that decide where the output goes and check that it arrived.

Background (notes/log.md, 13 Nov 22:00): a notebook written for Colab was run on Kaggle, whose image also has a
/content folder; it wrote its archives to /content/out, which Kaggle does not keep. These tests run the cells'
own code, including that case, and check that the final cell fails when an archive is missing.
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
    settings, setup, _clone, _env, _run, summary, download, final = _cells(gen.notebook(r, "0" * 40))
    ns = {}
    exec(settings, ns); exec(setup, ns)
    assert ns["PLATFORM"] == "local" and Path(ns["WORK"]) == clean_env / "kaggle_out"
    # what the runner leaves behind
    ns["REPO_DIR"] = str(clean_env / "repo"); (clean_env / "repo" / "results").mkdir(parents=True)
    run = ns["RUN_NAME"]
    status = {"planned": ["r2.qwen1.5b.cache"], "stages": {"r2.qwen1.5b.cache": {"ok": True, "minutes": 9.0, "detail": "ok"}},
              "models_run": ["qwen1.5b"], "export": {"zip_megabytes": {f"cache_{run}_qwen1.5b.zip": 434.8}}, "finished": True}
    (clean_env / "repo" / "results" / f"round2_status_{run}.json").write_text(json.dumps(status))
    for z in (f"results_{run}.zip", f"cache_{run}_qwen1.5b.zip"):
        _zip(Path(ns["WORK"]) / z)
    exec(summary, ns); exec(download, ns); exec(final, ns)
    assert ns["expected"] == [f"results_{run}.zip", f"cache_{run}_qwen1.5b.zip"] and ns["problems"] == []
    assert "OUTPUT CHECK PASSED" in capsys.readouterr().out
    # the activation archive is not there: the last cell must raise
    (Path(ns["WORK"]) / f"cache_{run}_qwen1.5b.zip").unlink()
    exec(summary, ns)
    with pytest.raises(RuntimeError, match="OUTPUT CHECK FAILED"):
        exec(final, ns)
    # the runner never reached its export step: also a failure, even with the small archive present
    status.pop("finished")
    (clean_env / "repo" / "results" / f"round2_status_{run}.json").write_text(json.dumps(status))
    exec(summary, ns)
    with pytest.raises(RuntimeError, match="did not reach its export step"):
        exec(final, ns)


@pytest.mark.skipif(os.path.isdir("/kaggle") or os.path.isdir("/content"), reason="written for a machine that is neither Kaggle nor Colab")
def test_gated_model_without_token_stops_in_the_first_cell(clean_env):
    r = next(x for x in gen.RUNS if x["file"] == "kaggle_r2_gemma.ipynb")
    settings, setup = _cells(gen.notebook(r, "0" * 40))[:2]
    ns = {}
    exec(settings, ns)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        exec(setup, ns)
    os.environ["HF_TOKEN"] = "hf_test_value_not_a_real_token"
    exec(setup, ns)                                                    # with a token in the environment it goes through
    assert "_tok" not in ns


def test_every_notebook_ends_with_the_output_check():
    nbs = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
    assert len(nbs) >= 9
    for p in nbs:
        cells = _cells(json.loads(p.read_text()))
        assert "OUTPUT CHECK FAILED" in cells[-1] and "raise RuntimeError" in cells[-1], p.name
        assert "choose_dirs(os.environ, os.path.isdir)" in cells[1], p.name
        assert '"/content/out" if' not in "".join(cells) and 'WORK = "/kaggle/working" if' not in "".join(cells), p.name
        for i, c in enumerate(cells):
            compile(c, f"{p.name}:cell{i}", "exec")
    assert callable(make_kaggle_notebook.main)
