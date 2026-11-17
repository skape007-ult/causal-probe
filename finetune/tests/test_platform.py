"""The platform decision of every fine-tuning notebook (17 Nov: colab_base on Colab was taken for Kaggle because
Colab's image has a /kaggle folder). Tests the function as it stands in ft_cells.py AND as written into each
committed notebook, so a notebook that was not regenerated fails here."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "finetune" / "scripts"))
from ft_cells import FT_PLATFORM_SETUP          # noqa: E402

NOTEBOOKS = sorted((ROOT / "finetune" / "notebooks").rglob("*.ipynb"))


def load_choose(cell_source):
    start, end = cell_source.index("def has_module"), cell_source.index("def under(")
    ns = {"os": os}
    exec(cell_source[start:end], ns)
    return ns["choose_dirs"]


def platform_cell(nb_path):
    nb = json.loads(nb_path.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    hits = [c for c in cells if "def choose_dirs" in c]
    assert len(hits) == 1, f"{nb_path.name}: expected one platform cell"
    return hits[0]


SOURCES = [("ft_cells", FT_PLATFORM_SETUP)] + [(p.relative_to(ROOT / "finetune" / "notebooks").as_posix(), None)
                                                  for p in NOTEBOOKS]


@pytest.fixture(params=SOURCES, ids=[s[0] for s in SOURCES])
def choose(request):
    name, src = request.param
    return load_choose(src if src is not None else platform_cell(ROOT / "finetune" / "notebooks" / name))


def folders(*present):
    return lambda p: p in present


def modules(*present):
    return lambda m: m in present


BOTH = folders("/content", "/kaggle", "/kaggle/working")


def test_colab_with_kaggle_folder_is_colab(choose):
    """The 17 Nov failure: Colab variables, google.colab importable, and a /kaggle folder present."""
    env = {"COLAB_RELEASE_TAG": "release-colab_20261110", "COLAB_GPU": "1"}
    assert choose(env, BOTH, modules("google.colab"))[:2] == ("colab", "/content/out")


@pytest.mark.parametrize("env", [{"COLAB_RELEASE_TAG": "x"}, {"COLAB_GPU": "1"}, {}])
def test_each_colab_signal_alone_beats_the_kaggle_folder(choose, env):
    mods = modules() if env else modules("google.colab")      # {} -> only the module says Colab
    assert choose(env, BOTH, mods)[0] == "colab"


@pytest.mark.parametrize("var", ["KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE"])
def test_kaggle_variable_decides_kaggle_even_on_a_colab_based_image(choose, var):
    """Kaggle's image is built on Colab's, so it may carry google.colab and a /content folder."""
    env = {var: "Batch", "COLAB_RELEASE_TAG": "x"}
    assert choose(env, BOTH, modules("google.colab")) == ("kaggle", "/kaggle/working/kaggle_out", "/kaggle/temp",
                                                           "/kaggle/working")


def test_kaggle_folder_alone_never_beats_colab(choose):
    assert choose({}, folders("/kaggle"), modules("google.colab"))[0] == "colab"


def test_folder_fallback(choose):
    assert choose({}, folders("/content"), modules())[0] == "colab"
    assert choose({}, folders("/kaggle"), modules())[0] == "kaggle"
    assert choose({}, folders(), modules())[0] == "local"


def test_both_folders_and_no_variable_refuses(choose):
    with pytest.raises(RuntimeError, match="platform unknown"):
        choose({}, BOTH, modules())


def test_every_notebook_states_its_platform_and_stops_on_mismatch():
    assert NOTEBOOKS, "no notebooks found"
    for p in NOTEBOOKS:
        nb = json.loads(p.read_text())
        text = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        if p.name == "ft_gputest.ipynb":
            continue                                    # path test: runs wherever it is launched
        want = "colab" if p.parent.name == "colab" else "kaggle"
        assert f'EXPECT_PLATFORM = "{want}"' in text, p.name
        assert "PLATFORM != EXPECT_PLATFORM" in text, p.name
        assert 'isdir("/kaggle"):' not in text.split("def choose_dirs")[1].split("def under(")[0].split("COLAB_VARS")[0], p.name
