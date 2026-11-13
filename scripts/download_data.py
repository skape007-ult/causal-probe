"""Download the two benchmarks at pinned revisions and verify their checksums.

  Corr2Cause  https://huggingface.co/datasets/causal-nlp/corr2cause   (Jin et al., arXiv 2306.05836)
  CLadder     https://huggingface.co/datasets/causal-nlp/CLadder      (Jin et al., arXiv 2312.04350)

These are the exact files every number in this repository was computed from. Safe to re-run.
"""
import hashlib
import shutil

from huggingface_hub import hf_hub_download

from causal_probe.config import DATA_RAW

FILES = [  # (repo, revision, file in repo, local path under data/raw, md5)
    ("causal-nlp/corr2cause", "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53", "train.csv", "corr2cause/train.csv", "bee2c98caa50f69bee54318d35f0ace4"),
    ("causal-nlp/corr2cause", "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53", "dev.csv", "corr2cause/dev.csv", "7a343e9d341b892a7a7b1b1bddd39aa9"),
    ("causal-nlp/corr2cause", "42ba12c769e11ff6427c9f52d7db58e3f9bf3e53", "test.csv", "corr2cause/test.csv", "c318d19ddc585bb4310084e2ce2d241f"),
    ("causal-nlp/CLadder", "18d684e444ed07cc57ae0f3a91ee983a79b8119a", "data/full_v1.5_default.csv", "cladder/full_v1.5_default.csv", "32815a3e44cbd32fa3db84972d750815"),
]


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    for repo, rev, name, local, want in FILES:
        dst = DATA_RAW / local
        if not (dst.exists() and md5(dst) == want):
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = hf_hub_download(repo, name, repo_type="dataset", revision=rev)
            shutil.copyfile(src, dst)
        got = md5(dst)
        assert got == want, f"{local}: md5 {got} != expected {want} (the upstream file changed?)"
        print(f"ok  {local}  {repo}@{rev[:8]}  md5 {got}")


if __name__ == "__main__":
    main()
