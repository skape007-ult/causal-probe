"""Generate every table used in README.md / report/report.md from results/*.json.

Writes report/tables/<name>.md. Each table ends with a line naming its source files, so every
number in the write-up traces to a JSON file. Nothing here is typed by hand.
"""
import json

from causal_probe.config import ALL_MODELS, MODELS, RESULTS, ROOT

OUT = ROOT / "report" / "tables"
LAB = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B", "gemma2b": "Gemma-2-2B", "qwen0.5b-lora": "Qwen2.5-0.5B + LoRA"}
DS = {"c2c_test": "Corr2Cause test", "c2c_probe": "Corr2Cause pool", "cladder": "CLadder"}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def ci(v, d=3, signed=False):
    if v is None or "est" not in v:
        return "–"
    f = f"{{:{'+' if signed else ''}.{d}f}}"
    if v.get("lo") is None:
        return f.format(v["est"])
    return f"{f.format(v['est'])} [{f.format(v['lo'])}, {f.format(v['hi'])}]"


def num(x, d=3, signed=False):
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    return f"{x:{'+' if signed else ''}.{d}f}"


def table(header, rows, sources):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    lines += ["", f"<sub>Source: {', '.join(f'`results/{s}`' for s in sources)}</sub>", ""]
    return "\n".join(lines)


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").write_text(text)
    print("wrote", name)


def t_behaviour(models=MODELS):
    rows, src = [], []
    for m in models:
        r = load(f"behaviour_{m}.json")
        if not r:
            continue
        src.append(f"behaviour_{m}.json")
        for tag in ("c2c_test", "cladder"):
            z = r["zero_shot"][tag]; s = z["logit_scored"]
            rows.append([LAB[m], DS[tag], z["n"], z["n_groups"], ci(s["accuracy"]), ci(s["macro_f1"]), ci(s["auroc_logit_diff"]),
                         num(s["frac_pred_yes"], 2), num(z["baselines"]["majority"]["accuracy"]),
                         num(z["baselines"]["majority"]["macro_f1"]), num(z["baselines"]["random"]["macro_f1"])])
    write("behaviour_zero_shot", table(
        ["Model", "Dataset", "n", "clusters", "Accuracy", "Macro-F1", "AUROC of logit diff", "Pred. Yes", "Majority acc.",
         "Majority macro-F1", "Random macro-F1"], rows, src))

    rows = []
    for m in models:
        r = load(f"behaviour_{m}.json")
        if not r:
            continue
        for tag, f in r["few_shot"].items():
            rows.append([LAB[m], DS[tag.removesuffix("_sub_fewshot")], f["n"], ci(f["zero_shot_same_items"]["accuracy"]),
                         ci(f["logit_scored"]["accuracy"]), ci(f["delta_accuracy_fewshot_minus_zeroshot"], signed=True),
                         ci(f["zero_shot_same_items"]["macro_f1"]), ci(f["logit_scored"]["macro_f1"]),
                         num(f["logit_scored"]["frac_pred_yes"], 2)])
    write("behaviour_few_shot", table(
        ["Model", "Dataset", "n", "Zero-shot acc.", "4-shot acc.", "Δ acc.", "Zero-shot macro-F1", "4-shot macro-F1", "4-shot pred. Yes"],
        rows, src))

    rows = []
    for m in models:
        r = load(f"behaviour_{m}.json")
        if not r:
            continue
        for tag, g in r["generation"].items():
            if g:
                rows.append([LAB[m], tag, g["n"], ci(g["logit_scored_same_items"]["accuracy"]), ci(g["gen_scored"]["accuracy"]),
                             num(g["logit_vs_gen_disagree_frac"]), num(g["unparseable_frac"])])
    write("behaviour_generation", table(
        ["Model", "Item set", "n", "Logit-scored acc.", "Generated-text acc.", "Disagreement", "Unparseable"], rows, src))

    rows = []
    for m in models:
        r = load(f"behaviour_{m}.json")
        if not r:
            continue
        b = r["robustness_variable_renaming"]
        rows.append([LAB[m], ci(b["original"]["macro_f1"]), ci(b["renamed"]["macro_f1"]),
                     ci(b["delta_macro_f1_renamed_minus_original"], signed=True), ci(b["delta_accuracy_renamed_minus_original"], signed=True),
                     ci(b["prediction_flip_frac"]), num(b["logit_diff_pearson_r"], 2)])
    write("robustness_renaming", table(
        ["Model", "Macro-F1 original", "Macro-F1 renamed", "Δ macro-F1", "Δ accuracy", "Predictions flipped", "r(logit diff)"], rows, src))

    for tag, key in (("cladder", "rung"), ("cladder", "query_type"), ("c2c_test", "template"), ("c2c_test", "num_variables")):
        rows = []
        vals = None
        for m in models:
            r = load(f"behaviour_{m}.json")
            if not r:
                continue
            b = r["zero_shot"][tag]["by"][key]
            vals = vals or list(b)
            rows.append([LAB[m]] + [ci(b[v]["accuracy"], 2) for v in vals])
        if rows:
            r0 = load(f"behaviour_{list(models)[0]}.json")["zero_shot"][tag]["by"][key]
            rows.append(["*n / frac. gold Yes*"] + [f"{r0[v]['n']} / {r0[v]['frac_gold_yes']:.2f}" for v in vals])
            write(f"behaviour_by_{tag}_{key}", table(["Model"] + [f"{key}={v}" for v in vals], rows, src))


def t_probes(models=MODELS, name="probes_summary"):
    rows, src = [], []
    for m in models:
        for tag in ("c2c_probe", "cladder"):
            c = load(f"confound_{m}_{tag}.json")
            if not c:
                continue
            p = load(f"probes_{m}_{tag}_C{c['selected_C_gold']:g}.json")
            src += [f"confound_{m}_{tag}.json", f"probes_{m}_{tag}_C{c['selected_C_gold']:g}.json"]
            L = c["selected_layer_gold"]; a = c["all_items"]
            lay = p["layers"]
            rows.append([LAB[m], DS[tag], f"{L}/{p['n_layers_plus_embed'] - 1}", f"{c['selected_C_gold']:g}",
                         ci(a["gold_probe"]["accuracy"]), ci(a["tfidf"]["accuracy"]), ci(a["type_prior"]["accuracy"]),
                         ci(a.get("random_init_probe", {}).get("accuracy")),
                         num(lay["shuffled"][L]["accuracy"]["est"]), num(lay["control"][L]["accuracy"]["est"]),
                         num(lay["gold"][L]["accuracy"]["est"] - lay["control"][L]["accuracy"]["est"], signed=True),
                         ci(a["gold_probe_minus_tfidf"]["accuracy"], signed=True)])
    write(name, table(
        ["Model", "Dataset", "Layer", "C", "Gold probe acc.", "TF-IDF acc.", "Type prior acc.", "Random-init probe acc.",
         "Shuffled labels", "Control task", "Selectivity", "Probe − TF-IDF"], rows, sorted(set(src))))


def t_confound(models=MODELS, suffix="", name="confound"):
    rows, rows2, src = [], [], []
    for m in models:
        for tag in ("c2c_probe", "cladder"):
            c = load(f"confound_{m}_{tag}{suffix}.json")
            if not c:
                continue
            src.append(f"confound_{m}_{tag}{suffix}.json")
            for label, root in (("raw", c), ("calibrated", c["calibrated"])):
                w = root["model_wrong"]
                if "gold_probe" not in w:
                    rows.append([LAB[m], DS[tag], label, w["n"]] + ["–"] * 6)
                    continue
                v = c["verdict"] if label == "raw" else c["verdict"]["calibrated"]
                rows.append([LAB[m], DS[tag], label, w["n"], ci(w["gold_probe"]["accuracy"]), ci(w["tfidf"]["accuracy"]),
                             ci(w["type_prior"]["accuracy"]), ci(w["gold_probe_minus_tfidf"]["accuracy"], signed=True),
                             ci(w["gold_probe_minus_type_prior"]["accuracy"], signed=True),
                             "**yes**" if v["represents_but_doesnt_say"] else "no"])
                for sname in ("model_said_yes", "model_said_no"):
                    b = root[sname]
                    if "gold_probe" not in b:
                        continue
                    rows2.append([LAB[m], DS[tag], label, sname.replace("model_said_", "said "), b["n"], ci(b["gold_probe"]["auroc"]),
                                  ci(b["tfidf"]["auroc"]), ci(b["type_prior"]["auroc"]), ci(b["model_logit_diff"]["auroc"]),
                                  ci(b["gold_probe_minus_tfidf"]["auroc"], signed=True)])
    write(f"{name}_wrong_items", table(
        ["Model", "Dataset", "Model answer", "n wrong", "Gold probe acc.", "TF-IDF acc.", "Type prior acc.", "Probe − TF-IDF",
         "Probe − type prior", "Criterion met"], rows, src))
    write(f"{name}_within_strata", table(
        ["Model", "Dataset", "Model answer", "Stratum", "n", "Gold probe AUROC", "TF-IDF AUROC", "Type prior AUROC",
         "Model logit-diff AUROC", "Probe − TF-IDF"], rows2, src))


def t_external(models=MODELS):
    rows, src = [], []
    for m in models:
        c = load(f"confound_{m}_c2c_probe.json")
        if not c or "external_c2c_test" not in c:
            continue
        src.append(f"confound_{m}_c2c_probe.json")
        e = c["external_c2c_test"]["all_items"]
        rows.append([LAB[m], e["n"], ci(e["gold_probe"]["auroc"]), ci(e["tfidf"]["auroc"]), ci(e["type_prior"]["auroc"]),
                     ci(e["model_logit_diff"]["auroc"]), ci(e["gold_probe_minus_tfidf"]["auroc"], signed=True)])
    write("external_c2c_test", table(
        ["Model", "n", "Gold probe AUROC", "TF-IDF AUROC", "Type prior AUROC", "Model logit-diff AUROC", "Probe − TF-IDF"], rows, src))


def t_steering():
    rows, src = [], []
    for m in MODELS:
        for tag in ("c2c_probe", "cladder"):
            r = load(f"steering_{m}_{tag}.json")
            if not r:
                continue
            src.append(f"steering_{m}_{tag}.json")
            for d, lab in (("gold_probe", "gold probe"), ("gold_diffmeans", "gold diff-of-means"), ("model_out", "model output")):
                x = r["directions"][d]
                a1 = next(v for v in x["add"] if v["alpha"] == 1.0); a4 = next(v for v in x["add"] if v["alpha"] == 4.0)
                rows.append([LAB[m], DS[tag], r["layer"], lab, ci(a1["flip_rate_plus"], 2), ci(a4["flip_rate_plus"], 2),
                             ci(a4["accuracy_toward_gold"], 2), num(x["ablate"]["mean_abs_delta_logit_diff"]),
                             num(r["direction_geometry"]["proj_std_over_s_unit"][d])])
            x = r["directions"]["random_mean"]
            a1 = next(v for v in x["add"] if v["alpha"] == 1.0); a4 = next(v for v in x["add"] if v["alpha"] == 4.0)
            rows.append([LAB[m], DS[tag], r["layer"], "random (mean of 3)", num(a1["flip_rate_plus"], 2), num(a4["flip_rate_plus"], 2),
                         num(a4["accuracy_toward_gold"], 2), num(x["ablate"]["mean_abs_delta_logit_diff"]), "–"])
    write("steering", table(
        ["Model", "Dataset", "Layer", "Direction", "Flip rate α=1", "Flip rate α=4", "Acc. towards gold α=4",
         "Ablation mean |ΔLD|", "Proj. std / s"], rows, src))


def t_patching():
    rows, rows2, src = [], [], []
    for m in MODELS:
        r = load(f"patching_{m}.json")
        if not r:
            continue
        src.append(f"patching_{m}.json")
        for kind in ("hyp", "prem"):
            k = r["kinds"][kind]; b = k["behaviour_on_pairs"]
            import numpy as np
            rec = np.array(k["pooled_recovery"], float); ctrl = np.array(k["control_pooled_recovery"], float)
            fin = rec[:, -1]
            first = next((r["layers"][i] for i, v in enumerate(fin) if v >= 0.5), None)
            rows.append([LAB[m], kind, k["n_pairs"], ci(b["mean_delta"], signed=True), ci(b["frac_delta_positive"], 2),
                         num(b["frac_both_correct"], 2), first if first is not None else "–",
                         f"{first / r['layers'][-1]:.2f}" if first is not None else "–",
                         num(float(np.nanmax(np.abs(ctrl[:, 0]))), 2), num(float(np.nanmax(np.abs(ctrl[:, -1]))), 2),
                         k["n_pairs_with_control"]])
            c = k.get("components", {})
            if "validated" in c:
                top = sorted(c["validated"], key=lambda v: -abs(v["real_patching"]))[:5]
                rows2.append([LAB[m], kind, c["n_pairs"], num(c["pearson_attr_vs_real"], 2), num(c["spearman_attr_vs_real"], 2),
                              num(c["mean_abs_real_top"]), num(c["mean_abs_real_random"]),
                              ", ".join(f"{v['component']} ({v['real_patching']:+.2f})" for v in top)])
            elif "error" in c:
                rows2.append([LAB[m], kind, "–", "–", "–", "–", "–", "failed: " + c["error"][:80]])
    write("patching_summary", table(
        ["Model", "Pair kind", "Pairs", "Mean D = LD(clean) − LD(corrupt)", "Frac. D > 0", "Both answered correctly",
         "First layer with ≥ 0.5 recovery at final token", "…as relative depth", "Max |control| at diff tokens",
         "Max |control| at final token", "Pairs with control"], rows, src))
    write("patching_components", table(
        ["Model", "Pair kind", "Pairs", "Pearson r (attr. vs real)", "Spearman", "Mean |real| top-10", "Mean |real| random-10",
         "Largest real effects"], rows2, src))


def t_cross_model():
    r = load("cross_model.json")
    if not r:
        return
    rows = []
    for m, row in r["models"].items():
        for tag, d in row["datasets"].items():
            rows.append([LAB[m], row["params_billion"], DS[tag], num(d["model_accuracy"]), ci(d["gold_probe_accuracy"]),
                         ci(d["probe_minus_tfidf_accuracy"], signed=True), ci(d.get("probe_minus_random_init_accuracy"), signed=True),
                         f"{d['selected_layer']}/{d['n_layers']} ({d['selected_layer_relative_depth']:.2f})",
                         num(d["first_layer_within_1pt_relative_depth"], 2), num(d["selectivity_at_selected_layer"], signed=True)])
    txt = table(["Model", "Params (B)", "Dataset", "Model acc.", "Gold probe acc.", "Probe − TF-IDF", "Probe − random-init",
                 "Selected layer (rel. depth)", "First layer within 1 pt (rel.)", "Selectivity"], rows, ["cross_model.json"])
    rows = [[k, ci(v, signed=True)] for k, v in r["pairwise"].items()]
    txt += "\n" + table(["Comparison (gold-probe accuracy, paired)", "Difference"], rows, ["cross_model.json"])
    write("cross_model", txt)


if __name__ == "__main__":
    t_behaviour(); t_probes(); t_confound(); t_confound(suffix="_fixedC", name="confound_fixedC"); t_external()
    t_steering(); t_patching(); t_cross_model()
