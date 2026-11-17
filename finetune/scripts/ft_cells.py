"""Notebook cells of the fine-tuning study that differ from v1.0's (v1.0's scripts/notebook_cells.py is not edited).

FT_PLATFORM_SETUP is v1.0's PLATFORM_SETUP with the platform decision replaced. Why (17 Nov): colab_base ran on Colab
but was taken for Kaggle, because Colab's image now has a /kaggle folder and v1.0 tested isdir("/kaggle") first. The
output went to /kaggle/working on the Colab VM and the browser download was skipped.

The decision, in this order:
  1. Kaggle only from Kaggle's own run variables: KAGGLE_KERNEL_RUN_TYPE or KAGGLE_URL_BASE.
  2. Colab from COLAB_RELEASE_TAG, COLAB_GPU or an importable google.colab.
  3. Only then folders: /content alone -> Colab, /kaggle alone -> Kaggle, neither -> local. Both folders and no
     variable -> stop with an error before any GPU time (both images now carry both folders, so they decide nothing).
Then each notebook states the platform it was written for (EXPECT_PLATFORM); a mismatch stops the run at once.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from notebook_cells import PLATFORM_SETUP          # noqa: E402

OLD_CHOOSE = '''def choose_dirs(env, isdir):
    """-> platform, output directory, scratch directory, the path this platform keeps (None: no rule)"""
    if env.get("KAGGLE_KERNEL_RUN_TYPE") or isdir("/kaggle"):
        return "kaggle", "/kaggle/working/kaggle_out", "/kaggle/temp", "/kaggle/working"
    if env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU") or isdir("/content"):
        return "colab", "/content/out", "/content/tmp", "/content/out"
    return "local", os.path.abspath("kaggle_out"), os.path.abspath("kaggle_tmp"), None
'''
OLD_CALL = "PLATFORM, WORK, TEMP, PERSIST = choose_dirs(os.environ, os.path.isdir)\n"
OLD_COMMENT = '''# archives are downloaded through the browser from /content/out. Kaggle's image also has a /content folder,
# which is why Kaggle is tested first.
'''

NEW_COMMENT = '''# archives are downloaded through the browser from /content/out. Both images now carry both /kaggle and /content
# folders, so the decision comes from each platform's own variables first and from folders only as a last resort
# (fine-tuning study, 17 Nov: a Colab run was taken for Kaggle by the old folder test).
'''
CHOOSE_DIRS = '''def has_module(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

KAGGLE_VARS, COLAB_VARS = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE"), ("COLAB_RELEASE_TAG", "COLAB_GPU")
DIRS = {"kaggle": ("kaggle", "/kaggle/working/kaggle_out", "/kaggle/temp", "/kaggle/working"),
        "colab": ("colab", "/content/out", "/content/tmp", "/content/out")}

def choose_dirs(env, isdir, has_module=has_module):
    """-> platform, output directory, scratch directory, the path this platform keeps (None: no rule).
    1. Kaggle only from its run variables; 2. Colab from its variables or google.colab; 3. folders, only if exactly
    one of /content and /kaggle exists. Both folders and no variable: refuse rather than guess."""
    if any(env.get(v) for v in KAGGLE_VARS):
        return DIRS["kaggle"]
    if any(env.get(v) for v in COLAB_VARS) or has_module("google.colab"):
        return DIRS["colab"]
    content, kaggle = isdir("/content"), isdir("/kaggle")
    if content and kaggle:
        raise RuntimeError("platform unknown: no Kaggle or Colab variable, and both /content and /kaggle exist")
    if content:
        return DIRS["colab"]
    if kaggle:
        return DIRS["kaggle"]
    return "local", os.path.abspath("kaggle_out"), os.path.abspath("kaggle_tmp"), None

def platform_signals(env, isdir, has_module=has_module):
    """What the decision saw (names only; no values are printed)."""
    return {"vars": [v for v in KAGGLE_VARS + COLAB_VARS if env.get(v)], "google.colab": has_module("google.colab"),
            "folders": [d for d in ("/content", "/kaggle") if isdir(d)]}
'''
NEW_CALL = '''PLATFORM, WORK, TEMP, PERSIST = choose_dirs(os.environ, os.path.isdir)
print("platform signals:", platform_signals(os.environ, os.path.isdir))
if globals().get("EXPECT_PLATFORM") and PLATFORM != EXPECT_PLATFORM:
    raise RuntimeError(f"this notebook is for {EXPECT_PLATFORM} but the platform was detected as {PLATFORM}; "
                       "nothing has run. Tell Claude the 'platform signals' line above.")
'''


def _swap(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("v1.0 PLATFORM_SETUP changed shape; cannot patch the platform decision safely")
    return text.replace(old, new)


FT_PLATFORM_SETUP = _swap(_swap(_swap(PLATFORM_SETUP, OLD_CHOOSE, CHOOSE_DIRS), OLD_CALL, NEW_CALL),
                          OLD_COMMENT, NEW_COMMENT)
