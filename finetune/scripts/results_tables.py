"""Number tables for finetune/RESULTS_FT.md, regenerated from a behaviour result file (no wording is chosen here).

  ../causal-probe/.venv/bin/python finetune/scripts/results_tables.py [finetune/results/behaviour_registered.json]
    -> finetune/results/<same name>_tables.md

Every number on the page is copied from this file's output, so that the audit can check page against file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["E2c", "E2a", "E2b", "E3", "E4a", "E4b", "E5a_A", "E5a_B", "E5b_A", "E5b_B", "E5c_A", "E5c_B"]
WHAT = {"E2c": "S(Bswap) - 0.5, POOL-LB (control)", "E2a": "S(B) - S(A), POOL-LB", "E2b": "S(B) - S(A48), POOL-LB",
        "E3": "S(A) - S(C), ID48", "E4a": "S(original) - S(renamed Z..U), A, ID48", "E4b": "S(keep order) - S(reverse), A, ID48",
        "E5a_A": "S in cells - 0.5, A", "E5a_B": "S in cells - 0.5, B", "E5b_A": "S - memorisation ref., A",
        "E5b_B": "S - memorisation ref., B", "E5c_A": "S - best text ref., A", "E5c_B": "S - best text ref., B"}


def f(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}" if isinstance(x, float) else str(x)


def main(path):
    path = Path(path).resolve()
    r = json.loads(path.read_text())
    T, D = r["tests"], r["descriptive"]
    L = [f"# Number tables from `{Path(path).relative_to(ROOT)}`", "",
         f"Mode: {r['meta']['mode']}; within-string implementation: {r['meta']['within_string_implementation']}; "
         f"premise resamples: {r['meta']['n_boot']}; written {r['meta']['written']}.", "",
         "## Registered tests (Holm over the eleven; E2c is a control, not in the family)", "",
         "| Test | Comparison | D | 95% interval | p | Holm p | Seed SD (x / y) | Premise SE | Outlying seeds | Claimed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ORDER:
        if k not in T:
            L.append(f"| {k} | {WHAT[k]} | not run | | | | | | | |"); continue
        t = T[k]
        out = (t["outlying_seeds_x"] + t["outlying_seeds_y"]) or "none"
        claim = t.get("direction", t.get("reading", ""))
        L.append(f"| {k} | {WHAT[k]} | {f(t['D'])} | [{f(t['lo'])}, {f(t['hi'])}] | {t['p_two_sided']:.4f} | "
                 f"{t['holm_p']:.4f} | {f(t['seed_sd_x'])} / {f(t['seed_sd_y'])} | {t['premise_se']:.3f} | {out} | {claim} |"
                 if "holm_p" in t else
                 f"| {k} | {WHAT[k]} | {f(t['D'])} | [{f(t['lo'])}, {f(t['hi'])}] | {t['p_two_sided']:.4f} | n/a | "
                 f"{f(t['seed_sd_x'])} | {t['premise_se']:.3f} | {out} | {claim} |")
        if "without_outliers" in t and "D" in t["without_outliers"]:
            w = t["without_outliers"]
            L.append(f"| {k} without outlying seed | | {f(w['D'])} | [{f(w['lo'])}, {f(w['hi'])}] | {w['p_two_sided']:.4f} | "
                     f"{f(t.get('holm_p_without_outliers'))} | | | | |")
    L += ["", "Power notes: " + ("; ".join(f"{k}: {n}" for k in ORDER if k in T for n in T[k].get("power_notes", [])) or "none"), ""]
    e1 = D["E1"]
    L += ["## E1 (gate, A)", "", f"S on ID48, mean over seeds: {e1['S_id48_mean']:.3f} (threshold {e1['thresholds']['S_id48']}); "
          f"per seed {', '.join(f'{x:.3f}' for x in e1['S_id48_per_seed'])}",
          f"Pooled AUROC on ID, mean: {e1['pooled_auroc_id_mean']:.3f} (threshold {e1['thresholds']['pooled_auroc_id']}). "
          f"Memorisation reference of A on ID48: {e1['memorisation_reference_A_id48']:.3f}. Passed: {e1['passed']}.", "",
          "## Rungs", "", f"A: {D['rung_A']}  ", f"B: {D['rung_B']}", "",
          "## S per model", "", "| Model | POOL-LB | ID48 |", "|---|---|---|"]
    for m, v in sorted(r["S_per_model"].items()):
        L.append(f"| {m} | {v.get('pool_lb', float('nan')):.3f} | {v.get('id48', float('nan')):.3f} |")
    for c in ("A", "B"):
        k = f"E5_near_far_{c}"
        if k in D:
            L += ["", f"## E5 descriptive, {c}: near and far halves of POOL-LB", "", "| Half | Items | S per seed | Memorisation ref. |", "|---|---|---|---|"]
            for h, v in D[k].items():
                L.append(f"| {h} | {v['n_items']} | {', '.join(f'{x:.3f}' for x in v['S_per_seed'])} | {v['memorisation_reference_S']:.3f} |")
        k = f"E5b_{c}_against_pair_only_variant"
        if k in D:
            t = D[k]
            L.append(f"\nE5b({c}) against the reviewer's pair-only variant (Amendment 1): D {f(t['D'])} [{f(t['lo'])}, {f(t['hi'])}], p {t['p_two_sided']:.4f}")
    L += ["", "## Reference checks", "", "```", json.dumps(r["meta"]["reference_checks"], indent=1), "```"]
    out = Path(path).with_name(Path(path).stem + "_tables.md")
    out.write_text("\n".join(L) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "finetune" / "results" / "behaviour_registered.json")
