"""Notebook cells shared by make_kaggle_notebook.py and make_round2_notebooks.py.

Why this file exists (notes/log.md, 13 Nov 22:00): the Colab notebook for Qwen 1.5B was run on Kaggle.
Kaggle's image also has a /content folder, so the notebook wrote its archives to /content/out; Kaggle keeps
only /kaggle/working after "Save & Run All", and 72 minutes of GPU output were lost. Every notebook now
  1. chooses its output directory by platform (Kaggle is tested first),
  2. checks before any GPU time is spent that this directory is under the path the platform keeps,
  3. ends with a cell that lists the output directory, writes STATUS.txt (OK / FAILED / SKIPPED) into it and prints a banner.
     It does not raise: whether Kaggle keeps the output of a version whose last cell failed is not known, so the
     check must not depend on it. scripts/kaggle_ingest.py refuses a run whose STATUS.txt is missing or not OK.
tests/test_notebook_cells.py runs these cells' functions, including the case that went wrong.
"""

# Needs: os imported, NEEDS_TOKEN defined. Defines PLATFORM, WORK, TEMP, PERSIST.
PLATFORM_SETUP = '''
# ---- which platform is this, and where must the output go? --------------------
# Kaggle keeps ONLY /kaggle/working after "Save & Run All". Colab keeps nothing after the session, so the
# archives are downloaded through the browser from /content/out. Kaggle's image also has a /content folder,
# which is why Kaggle is tested first.
def choose_dirs(env, isdir):
    """-> platform, output directory, scratch directory, the path this platform keeps (None: no rule)"""
    if env.get("KAGGLE_KERNEL_RUN_TYPE") or isdir("/kaggle"):
        return "kaggle", "/kaggle/working/kaggle_out", "/kaggle/temp", "/kaggle/working"
    if env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU") or isdir("/content"):
        return "colab", "/content/out", "/content/tmp", "/content/out"
    return "local", os.path.abspath("kaggle_out"), os.path.abspath("kaggle_tmp"), None

def under(path, parent):
    return (os.path.abspath(path) + "/").startswith(os.path.abspath(parent) + "/")

PLATFORM, WORK, TEMP, PERSIST = choose_dirs(os.environ, os.path.isdir)
os.makedirs(TEMP, exist_ok=True); os.makedirs(WORK, exist_ok=True)
# Stop here, before any GPU time is spent, if the archives would be written where the platform does not keep them.
if PERSIST and not under(WORK, PERSIST):
    raise RuntimeError(f"output directory {WORK} is not under {PERSIST}: on {PLATFORM} the archives would be lost")
with open(WORK + "/.write_test", "w") as f:
    f.write("ok")
os.remove(WORK + "/.write_test")
print("platform:", PLATFORM, "| archives will be written to", WORK,
      {"kaggle": "(kept by Kaggle: Output tab of the saved version, folder kaggle_out)",
       "colab": "(NOT kept after the session: the download cell near the end hands them to the browser)"}.get(PLATFORM, ""))

# ---- Hugging Face token, from this platform's Secrets ---------------------------
# It is never printed or written to disk by this notebook. A secret is attached PER NOTEBOOK: on Kaggle,
# Add-ons -> Secrets -> tick HF_TOKEN for this notebook (a new or imported notebook starts with none attached).
def read_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "the environment"
    if PLATFORM == "kaggle":
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN"), "Kaggle Secrets"
    if PLATFORM == "colab":
        from google.colab import userdata
        return userdata.get("HF_TOKEN"), "Colab Secrets"
    return None, ""
try:
    _tok, _where = read_token()
    if not _tok:
        raise LookupError("empty")
    os.environ["HF_TOKEN"] = _tok
    del _tok
    print("HF_TOKEN loaded from", _where)
except Exception as e:
    print("No HF_TOKEN secret is attached to this notebook (%s)." % type(e).__name__)
    if NEEDS_TOKEN:
        raise RuntimeError("This run needs the HF_TOKEN secret (the Gemma weights are gated). Attach it to THIS notebook "
                           "(Kaggle: Add-ons -> Secrets -> tick HF_TOKEN; Colab: the key icon in the left bar) and run again.")
    print("This run does not need one: the Qwen weights are public and are downloaded anonymously.")
'''

# Needs: PLATFORM, WORK, zips (list of archive names in WORK).
COLAB_DOWNLOAD = '''
# ---- Colab only: hand the archives to the browser -------------------------------
# Colab deletes /content when the session ends, so download now. If the browser blocks the downloads,
# open the Files panel (folder icon, left), go to /content/out and download from there.
if PLATFORM == "colab":
    try:
        from google.colab import files
        for f in zips:
            files.download(WORK + "/" + f)
    except Exception as e:
        print("automatic download not available here:", type(e).__name__)
else:
    print("not on Colab: nothing to do in this cell")
'''

# Needs: os, PLATFORM, WORK, PERSIST, under(), COMMIT, RUN (the run's name), expected (list of archive names),
# not_finished (str or ""), skipped (str or ""). Defines OUTPUT_STATE.
FINAL_CHECK = '''
# ---- final check: are the archives where this platform keeps them? --------------
# Writes STATUS.txt into the output directory (first line: OK, FAILED or SKIPPED), prints a banner, and ends normally:
# it never raises, because a version whose last cell failed might not keep its output. The ingest on the laptop
# refuses a run whose STATUS.txt is missing or does not say OK.
import datetime, zipfile

def check_outputs(work, persist, expected):
    \"\"\"List the output directory with sizes. Returns the list of problems (empty: all expected archives are in place).\"\"\"
    problems, total = [], 0
    print("output directory:", work)
    for root, _, fs in sorted(os.walk(work)):
        for f in sorted(fs):
            p = os.path.join(root, f); n = os.path.getsize(p); total += n
            print(f"  {n / 1e6:10.1f} MB  {os.path.relpath(p, work)}")
    print(f"  {total / 1e6:10.1f} MB  in total")
    if persist and not under(work, persist):
        problems.append(f"the output directory {work} is not under {persist}, the only place this platform keeps")
    if not expected:
        problems.append("no archive was expected: the runner left no record of what it exported")
    for f in expected:
        p = os.path.join(work, f)
        if not os.path.isfile(p):
            problems.append("expected archive is missing: " + p)
        elif not zipfile.is_zipfile(p) or os.path.getsize(p) < 1000:
            problems.append("expected archive is empty or not a zip file: " + p)
    return problems

def write_status(work, state, run, commit, platform, expected, problems):
    \"\"\"STATUS.txt in the output directory, and a copy inside the small results archive (results/round2_output_status_<run>.txt),
    so that the status travels with the download even if STATUS.txt itself is not downloaded.\"\"\"
    lines = [state, "run: " + run, "commit: " + commit, "platform: " + platform, "output directory: " + work,
             "written: " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "archives:"]
    for f in expected:
        p = os.path.join(work, f)
        lines.append("  " + f + "  " + (str(os.path.getsize(p)) + " bytes" if os.path.isfile(p) else "MISSING"))
    lines.append("problems:" + ("" if problems else " none"))
    lines += ["  - " + q for q in problems]
    text = "\\n".join(lines) + "\\n"
    with open(os.path.join(work, "STATUS.txt"), "w") as f:
        f.write(text)
    member = "results/round2_output_status_" + run + ".txt"
    for f in expected:
        p = os.path.join(work, f)
        if f.startswith("results") and os.path.isfile(p) and zipfile.is_zipfile(p):
            tmp = p + ".tmp"                      # rewritten rather than appended to, so that re-running this cell replaces the copy
            with zipfile.ZipFile(p) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
                for it in zi.infolist():
                    if it.filename != member:
                        zo.writestr(it, zi.read(it.filename))
                zo.writestr(member, text)
            os.replace(tmp, p)
    return text

try:
    problems = ([not_finished] if not_finished and not skipped else []) + check_outputs(WORK, PERSIST, expected)
    OUTPUT_STATE = "SKIPPED" if skipped else ("FAILED" if problems else "OK")
    if skipped:
        problems = ["the run skipped itself: " + skipped] + problems
    status_text = write_status(WORK, OUTPUT_STATE, RUN, COMMIT, PLATFORM, expected, problems)
except Exception as e:                            # the check itself must not be what loses the output
    OUTPUT_STATE, problems, status_text = "FAILED", ["the output check itself failed: %s: %s" % (type(e).__name__, e)], ""
    try:
        with open(os.path.join(WORK, "STATUS.txt"), "w") as f:
            f.write("FAILED\\nrun: %s\\nproblems:\\n  - %s\\n" % (RUN, problems[0]))
    except Exception:
        pass
bar = ("#" if OUTPUT_STATE == "OK" else "!") * 100
print("\\n" + bar)
if OUTPUT_STATE == "OK":
    print("OUTPUT CHECK PASSED:", len(expected), "archive(s) and STATUS.txt in", WORK)
    print({"kaggle": "Download the zip files and STATUS.txt from the Output tab of this saved version (folder kaggle_out).",
           "colab": "The archives were handed to the browser by the cell above; if not, download them from the Files panel now."}.get(PLATFORM, ""))
elif OUTPUT_STATE == "SKIPPED":
    print("RUN SKIPPED, nothing to download. Reason: " + skipped)
else:
    print("OUTPUT CHECK FAILED. Save what is listed above by hand before this session ends. The ingest will refuse this run.")
for q in problems:
    print("  - " + q)
print(bar)
print("STATUS.txt, first line:", OUTPUT_STATE)
'''
