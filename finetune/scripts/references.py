"""References that need no model (PREREG E1, E3, E5), computed before any model is trained.
  caffeinate -i ../causal-probe/.venv/bin/python finetune/scripts/references.py    -> finetune/itemsets/references.json

For each training set (A, A48, B, Bswap, C) and the sets POOL-LB and ID48: the registered statistic S of every member
of the memorisation family and of the per-string premise-count classifier; the family's largest S is the
"memorisation reference" of that condition on that set. Also the string prior with and without length on ID.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "finetune")]
import numpy as np                                   # noqa: E402
from causal_probe import prompts                      # noqa: E402
from ftstudy import datasets as S, stat               # noqa: E402


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ntok = lambda items: np.array([len(prompts.encode_with_positions(tok, it)["input_ids"]) for it in items])
    ev = {k: S.materialise("eval_" + k) for k in ("pool_lb", "id48", "id")}
    meta = {k: (np.array([i["label"] for i in v]), np.array([i["query"] for i in v]), ntok(v)) for k, v in ev.items()}
    out = {"statistic": "S as registered (strings with at least 3 items per class; length removed)", "by_training_set": {}}
    for cond in ("A", "A48", "B", "Bswap", "C"):
        tr = S.materialise("train_" + cond)
        r = {}
        for k in ("pool_lb", "id48"):
            y, q, n = meta[k]
            fam = {name: stat.S(y, sc, q, n) for name, sc in S.memorisation_family(tr, ev[k]).items()}
            best = max(fam, key=fam.get)
            r[k] = {"memorisation_family": {a: round(b, 4) for a, b in fam.items()}, "memorisation_reference": {"member": best, "S": fam[best]},
                    "nearest_premise_k1_all_statements": fam["all|k|1"],
                    "premise_counts_within_string": {"logistic": stat.S(y, S.premise_counts_within_string(tr, ev[k]), q, n),
                                                     "boosting": stat.S(y, S.premise_counts_within_string(tr, ev[k], boosting=True), q, n)}}
        y, q, n = meta["id"]
        tn = ntok(tr)
        st = S.shortcut_transfer(tr, tn, ev["id"], n, n_boot=200)
        r["id"] = {"pooled_auroc_string_prior": st["hypothesis_only"]["auroc"], "pooled_auroc_string_and_length": st["string_and_length"]["auroc"]}
        out["by_training_set"][cond] = r
        print(cond, {k: (r[k]["memorisation_reference"]["member"], round(r[k]["memorisation_reference"]["S"], 3), round(r[k]["nearest_premise_k1_all_statements"], 3),
                         {a: round(b, 3) for a, b in r[k]["premise_counts_within_string"].items()}) for k in ("pool_lb", "id48")}, r["id"], flush=True)
    # a perfect validity-tracker's S after length removal (how much the removal costs a model that is right)
    for k in ("pool_lb", "id48"):
        y, q, n = meta[k]
        out.setdefault("S_of_the_label_itself_plus_noise", {})[k] = stat.S(y, y + 0.3 * np.random.default_rng(0).standard_normal(len(y)), q, n)
        out.setdefault("eligible_strings", {})[k] = int(len(np.unique(q[stat.eligible(y, q)])))
    (ROOT / "finetune" / "itemsets" / "references.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
