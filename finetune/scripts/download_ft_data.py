"""The authors' perturbed test sets, at the revision v1.0 pinned for Corr2Cause, verified by md5 (the checksums are
those of the files the data build on the laptop used). train/dev/test come from the v1.0 scripts/download_data.py."""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from causal_probe.config import DATA_RAW          # noqa: E402

REPO, REV = "causal-nlp/corr2cause", "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53"
FILES = {"perturbation_by_paraphrasing_test.csv": "ff9e823e8673888d6f1e9585ea618403",
         "perturbation_by_refactorization_test.csv": "32e1bbae1389d93ed11dd965974bba47"}


def main():
    from huggingface_hub import hf_hub_download
    for name, want in FILES.items():
        dst = DATA_RAW / "corr2cause" / name
        if not (dst.exists() and hashlib.md5(dst.read_bytes()).hexdigest() == want):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hf_hub_download(REPO, name, repo_type="dataset", revision=REV), dst)
        got = hashlib.md5(dst.read_bytes()).hexdigest()
        assert got == want, f"{name}: md5 {got} != expected {want}"
        print(f"ok  corr2cause/{name}  {REPO}@{REV[:8]}  md5 {got}")


if __name__ == "__main__":
    main()
