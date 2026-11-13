"""Generate every table used in README.md / report/report.md from results/*.json.

Writes report/tables/<name>.md. Each table ends with a line naming its source files, so every
number in the write-up traces to a JSON file. Nothing here is typed by hand.
"""
import json

from causal_probe.config import ALL_MODELS, MODELS, REMOTE_MODELS, RESULTS, ROOT

MODELS = {**MODELS, **REMOTE_MODELS}     # remote (Kaggle) models appear in a table once their results exist

import os
from pathlib import Path

OUT = Path(os.environ.get("CAUSAL_PROBE_TABLES", ROOT / "report" / "tables"))
LAB = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B", "gemma2b": "Gemma-2-2B", "qwen0.5b-lora": "Qwen2.5-0.5B + LoRA",
       "qwen3b": "Qwen2.5-3B", "qwen7b-4bit": "Qwen2.5-7B (4-bit)"}
DS = {"c2c_test": "Corr2Cause test", "c2c_probe": "Corr2Cause pool", "cladder": "CLadder",
      "c2c_debiased": "Corr2Cause debiased pool", "c2c_test_renamed": "Corr2Cause test, variables renamed",
      "c2c_test_sub_fewshot": "Corr2Cause test (500 items)", "cladder_sub_fewshot": "CLadder (500 items)"}


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
         "Shuffled labels (all items)", "Control task (all items)", "Selectivity (all items)", "Probe − TF-IDF"], rows, sorted(set(src))) +
        "\n<sub>Layer index counts the embeddings as 0, so layer l is the output of block l−1. Probe/TF-IDF/prior/random-init columns: report half, "
        "scores averaged over 3 seeds. Shuffled/control/selectivity: all items, mean of per-seed accuracies at the selected layer and C.</sub>\n")


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
                    rows.append([LAB[m], DS[tag], label, w["n"]] + ["–"] * 10)
                    continue
                v = c["verdict"] if label == "raw" else c["verdict"]["calibrated"]
                rows.append([LAB[m], DS[tag], label, w["n"], num(w.get("frac_gold_yes"), 2),
                             ci(w["gold_probe"].get("balanced_accuracy")),
                             ci(w["gold_probe"]["accuracy"]), ci(w["tfidf"]["accuracy"]),
                             ci(w["type_prior"]["accuracy"]), ci(w["gold_probe_minus_tfidf"]["accuracy"], signed=True),
                             ci(w["gold_probe_minus_type_prior"]["accuracy"], signed=True),
                             "**yes**" if v["represents_but_doesnt_say"] else "no",
                             ci(w.get("structural", {}).get("accuracy")),
                             ci(w.get("gold_probe_minus_structural", {}).get("accuracy"), signed=True)])
                for sname in ("model_said_yes", "model_said_no"):
                    b = root[sname]
                    if "gold_probe" not in b:
                        continue
                    rows2.append([LAB[m], DS[tag], label, sname.replace("model_said_", "said "), b["n"], ci(b["gold_probe"]["auroc"]),
                                  ci(b["tfidf"]["auroc"]), ci(b["type_prior"]["auroc"]), ci(b["model_logit_diff"]["auroc"]),
                                  ci(b["gold_probe_minus_tfidf"]["auroc"], signed=True),
                                  ci(b.get("structural", {}).get("auroc")),
                                  ci(b.get("gold_probe_minus_structural", {}).get("auroc"), signed=True)])
    write(f"{name}_wrong_items", table(
        ["Model", "Dataset", "Model answer", "n wrong", "Frac. gold-Yes among them", "Gold probe balanced acc.", "Gold probe acc.", "TF-IDF acc.", "Type prior acc.", "Probe − TF-IDF",
         "Probe − type prior", "Pre-registered criterion met", "Parsed features + prior acc. (post-hoc)", "Probe − parsed features (post-hoc)"], rows, src))
    write(f"{name}_within_strata", table(
        ["Model", "Dataset", "Model answer", "Stratum", "n", "Gold probe AUROC", "TF-IDF AUROC", "Type prior AUROC",
         "Model logit-diff AUROC", "Probe − TF-IDF", "Structural AUROC (post-hoc)", "Probe − structural (post-hoc)"], rows2, src))


def t_audited(models=MODELS):
    rows, rows2, src = [], [], []
    for m in models:
        for tag in ("c2c_probe", "cladder", "c2c_debiased"):
            c = load(f"confound_{m}_{tag}.json")
            if not c or "audited" not in c.get("verdict", {}):
                continue
            src.append(f"confound_{m}_{tag}.json")
            v = c["verdict"]["audited"]; a = c["all_items"]; tb = c["text_baselines_all_items"]
            st = v["calibrated_strata_probe_minus_best_text_auroc"]
            rows.append([LAB[m], DS[tag], ci(a["gold_probe"]["accuracy"]), v["best_text_baseline"].replace("_", " "),
                         ci(a["best_text"]["accuracy"]), ci(v["all_items_probe_minus_best_text_accuracy"], signed=True),
                         ci(st.get("model_said_yes"), signed=True), ci(st.get("model_said_no"), signed=True),
                         "**yes**" if v["represents_beyond_best_text_baseline"] else "no",
                         "yes" if c["verdict"]["represents_but_doesnt_say"] else "no",
                         "yes" if v["raw_wrong_items_criterion_applicable"] else f"no ({v['raw_wrong_items_frac_gold_yes']:.0%} gold-Yes)"])
            rows2.append([LAB[m], DS[tag], ci(a["gold_probe"]["accuracy"])] +
                         [ci(a[k]["accuracy"]) if k in a else "–" for k in ("tfidf", "type_prior", "tfidf_query", "hyp_prior", "structural")] +
                         [ci(a.get("random_init_probe", {}).get("accuracy")), ci(a["model_logit_diff"]["accuracy"])])
    write("audited_verdict", table(
        ["Model", "Dataset", "Gold probe acc.", "Strongest text baseline", "Its acc.", "Probe − baseline (acc., all items)",
         "Probe − baseline (AUROC, model leans Yes)", "Probe − baseline (AUROC, model leans No)", "Audited criterion met",
         "Pre-registered criterion met (as coded)", "Pre-registered wrong-items test applicable"], rows, src))
    write("text_baselines", table(
        ["Model", "Dataset", "Gold probe", "TF-IDF (full prompt)", "Template / type prior", "TF-IDF (question only)",
         "Hypothesis-string prior", "Parsed features + prior (GBM)", "Random-init probe", "Model's own answer"], rows2, src))
    rows = []
    for m in models:
        for tag in ("c2c_probe", "cladder", "c2c_debiased"):
            c = load(f"confound_{m}_{tag}.json")
            if not c or "content_end" not in c.get("positions", {}):
                continue
            a = c["all_items"]; ps = c["positions"]
            rows.append([LAB[m], DS[tag], ci(a["gold_probe"]["accuracy"]), ci(ps["content_end"]["accuracy"]),
                         ci(ps["premise_end"]["accuracy"]), ci(ps["premise_end"]["final_minus_this_accuracy"], signed=True),
                         ci(ps["content_end"]["final_minus_this_accuracy"], signed=True),
                         ci(a.get("random_init_probe", {}).get("accuracy")), ci(ps["content_end"].get("random_init_accuracy")),
                         ci(ps["content_end"].get("trained_minus_random_init_accuracy"), signed=True)])
    write("probe_positions", table(
        ["Model", "Dataset", "Final token", "End of hypothesis / question", "End of premise / context (question not yet read)",
         "Final − end of premise", "Final − end of hypothesis", "Random-init model, final token",
         "Random-init model, end of hypothesis", "Trained − random-init, end of hypothesis"], rows, src))


def t_external(models=MODELS):
    rows, src = [], []
    for m in models:
        c = load(f"confound_{m}_c2c_probe.json")
        if not c or "external_c2c_test" not in c:
            continue
        src.append(f"confound_{m}_c2c_probe.json")
        e = c["external_c2c_test"]["all_items"]
        rows.append([LAB[m], e["n"], ci(e["gold_probe"]["auroc"]), ci(e["tfidf"]["auroc"]), ci(e["type_prior"]["auroc"]),
                     ci(e["model_logit_diff"]["auroc"]), ci(e["gold_probe_minus_tfidf"]["auroc"], signed=True),
                     ci(e.get("hyp_prior", {}).get("auroc")), ci(e.get("gold_probe_minus_hyp_prior", {}).get("auroc"), signed=True),
                     ci(e.get("structural", {}).get("auroc")), ci(e.get("gold_probe_minus_structural", {}).get("auroc"), signed=True)])
    write("external_c2c_test", table(
        ["Model", "n", "Gold probe AUROC", "TF-IDF AUROC", "Type prior AUROC", "Model logit-diff AUROC", "Probe − TF-IDF",
         "Hypothesis-prior AUROC", "Probe − hypothesis prior", "Parsed features + prior AUROC", "Probe − parsed features"], rows, src))


def t_steering():
    """Steering tables from the audited re-analysis (07b) and the matched controls (07c)."""
    names = (("gold_probe", "gold-label probe weights"), ("gold_diffmeans", "gold difference of class means"),
             ("model_out", "model-answer probe weights (positive control)"))
    rows, rows2, rows3, src, src3 = [], [], [], [], []
    for m in MODELS:
        for tag, suffix in (("c2c_probe", ""), ("c2c_probe", "_d0.5"), ("c2c_probe", "_d0.85"), ("c2c_debiased", ""),
                            ("cladder", ""), ("cladder", "_d0.5"), ("cladder", "_d0.85")):
            r = load(f"steering_analysis_{m}_{tag}{suffix}.json")
            if not r:
                continue
            src.append(f"steering_analysis_{m}_{tag}{suffix}.json")
            site = f"{r['layer']}/{r['n_layers']}"
            dnames = list(names) + [(k, f"random direction {k[-1]}") for k in r["directions"] if k.startswith("random")]
            for d, lab in dnames:
                x = r["directions"][d]
                f4 = next(v for v in x["flip_rate"] if v["alpha"] == 4.0)
                rows.append([LAB[m], DS[tag], site, lab, ci(x["slope_per_s_at_alpha_1"], signed=True),
                             num(x["frac_items_with_positive_slope_at_alpha_1"], 2), num(x["alpha_1_in_own_sd"], 1),
                             num(x["slope_per_own_sd_at_alpha_1"], 4, signed=True),
                             num(x["effect_of_natural_class_gap_on_logit_diff"], 4, signed=True),
                             num(x["auroc_gold_report_half"]), num(x["auroc_model_answer_report_half"])])
                rows2.append([LAB[m], DS[tag], site, lab, ci(f4["plus"], 2), ci(f4["minus"], 2),
                              ci(f4["controllability_toward_gold"], 2), ci(f4["controllability_away_from_gold"], 2),
                              num(x["ablation"]["mean_abs_delta_logit_diff"], 4), num(r["ablation_noise_floor_logit_diff"], 4),
                              "yes" if x["ablation"]["above_noise_floor"] else "no"])
        for tag in ("c2c_probe", "c2c_debiased", "cladder"):
            c = load(f"steering_controls_{m}_{tag}.json")
            if not c:
                continue
            src3.append(f"steering_controls_{m}_{tag}.json")
            for d, lab in names:
                x = c["real"][d]; mf = x["matched_control_family"]; fam = c["families"]
                rows3.append([LAB[m], DS[tag], c["layer"], lab, ci(x["slope_per_s"], signed=True),
                              f"{100 * x['versus']['isotropic']['percentile_of_abs_slope_per_s']:.0f}",
                              num(fam["isotropic"]["abs_slope_per_s"]["median"]), num(fam["isotropic"]["abs_slope_per_s"]["max"]),
                              mf.replace("_", "-label ", 1).replace("diffmeans", "difference of means"),
                              f"{100 * x['versus'][mf]['percentile_of_abs_slope_per_s']:.0f}",
                              num(fam[mf]["abs_slope_per_s"]["median"]), num(fam[mf]["abs_slope_per_s"]["max"]),
                              num(x["versus"][mf]["ratio_to_median_abs_slope_per_s"], 1)])
    write("steering", table(
        ["Model", "Item set", "Layer", "Direction", "Slope: Δ(Yes−No logit) per unit s", "Share of items with positive slope",
         "One unit of s in the direction's own SDs", "Slope per own SD", "Effect of moving across the natural gold-class gap",
         "AUROC for gold label", "AUROC for model's answer"], rows, src))
    write("steering_flips", table(
        ["Model", "Item set", "Layer", "Direction", "Flip rate +4s", "Flip rate −4s", "Controllability towards gold (4s)",
         "Controllability away from gold (4s)", "Ablation mean |ΔLD|", "Noise floor", "Ablation above 3× floor"], rows2, src))
    if rows3:
        write("steering_controls", table(
            ["Model", "Item set", "Layer", "Direction", "Slope per unit s", "Percentile among 100 isotropic directions (|slope|)",
             "Isotropic median |slope|", "Isotropic max |slope|", "Matched control family (20 directions)",
             "Percentile among matched controls", "Matched median |slope|", "Matched max |slope|", "Ratio to matched median"], rows3, src3))


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
                         ci(d.get("probe_minus_best_text_accuracy"), signed=True), ci(d.get("probe_minus_random_init_accuracy"), signed=True),
                         f"{d['selected_layer']}/{d['n_layers']} ({d['selected_layer_relative_depth']:.2f})",
                         num(d["first_layer_within_1pt_relative_depth"], 2), num(d["selectivity_at_selected_layer"], signed=True)])
    txt = table(["Model", "Params (B)", "Dataset", "Model acc.", "Gold probe acc.", "Probe − strongest text baseline", "Probe − random-init",
                 "Selected layer (rel. depth)", "First layer within 1 pt (rel.)", "Selectivity"], rows, ["cross_model.json"])
    rows = [[k, ci(v, signed=True)] for k, v in r["pairwise"].items()]
    txt += "\n" + table(["Comparison (gold-probe accuracy, paired)", "Difference"], rows, ["cross_model.json"])
    write("cross_model", txt)


def t_benchmarks():
    r = load("benchmark_baselines.json")
    if not r:
        return
    src = ["benchmark_baselines.json"]
    names = {"majority": "Majority class", "template_prior": "Template prior (6 templates)",
             "hypothesis_only": "Hypothesis only (string prior)", "hypothesis_only_f1_threshold": "Hypothesis only, F1-tuned threshold",
             "premise_only": "Premise only (TF-IDF)", "parsed_features": "Parsed features + hypothesis prior (GBM)",
             "parsed_features_excluding_premises_seen_in_train": "Parsed features, dev premises also in train removed",
             "hypothesis_only_tfidf": "Hypothesis only (TF-IDF)", "full_text_tfidf": "Whole prompt (TF-IDF)",
             "type_prior": "Graph x query-type prior", "question_only": "Question only (TF-IDF)", "full_text": "Whole prompt (TF-IDF)",
             "numeric_polarity": "Quoted numbers + question polarity (GBM)"}
    rows = []
    for split in ("test", "dev"):
        for k, v in r["corr2cause_official_splits"][split].items():
            rows.append([split, names[k], v["n"], v["n_clusters"], ci(v["accuracy"]), ci(v["macro_f1"]), ci(v["positive_f1"]),
                         ci(v["auroc"]), num(v["frac_predicted_positive"], 2)])
    write("benchmark_c2c_official", table(
        ["Split", "Text-only baseline (fit on train)", "Items", "Premises", "Accuracy", "Macro-F1", "F1 (valid class)", "AUROC",
         "Predicted valid"], rows, src))
    rows = []
    for key, lab in (("corr2cause_balanced_pool", "Balanced pool"), ("corr2cause_debiased_pool", "Debiased pool"),
                     ("corr2cause_string_balanced_only_pool_rejected", "Rejected first design (strings balanced over the whole pool only)")):
        for k, v in r[key]["baselines"].items():
            rows.append([lab, names[k], v["n"], v["n_clusters"], ci(v["accuracy"]), ci(v["macro_f1"]), ci(v["auroc"])])
    write("benchmark_c2c_pools", table(
        ["Corr2Cause item set", "Text-only baseline (premise-held-out cross-validation)", "Items", "Premises", "Accuracy", "Macro-F1", "AUROC"],
        rows, src))
    d = r["corr2cause_description"]
    rows = [[sp, d[sp]["n_items"], d[sp]["n_distinct_premises"], num(d[sp]["items_per_premise_median"], 0), num(d[sp]["frac_valid"]),
             d[sp]["debiasing"]["n_hypothesis_strings"], d[sp]["debiasing"]["n_strings_single_label"],
             d[sp]["debiasing"]["max_items_balanced_within_string"]] for sp in ("train", "dev", "test")]
    txt = table(["Split", "Items", "Distinct premises", "Median items per premise", "Fraction valid", "Distinct hypothesis strings",
                 "Strings with a single label in this split", "Items left if labels are balanced within each string"], rows, src)
    o = d["train_validity_by_variable_name_order"]
    rows = [[t, v["example_hypothesis"], v["first_named_variable_earlier_in_alphabet"]["n"], num(v["first_named_variable_earlier_in_alphabet"]["frac_valid"]),
             v["first_named_variable_later_in_alphabet"]["n"], num(v["first_named_variable_later_in_alphabet"]["frac_valid"])] for t, v in o.items()]
    txt += "\n" + table(["Template (train)", "Example hypothesis", "Items, first-named variable earlier in alphabet", "Fraction valid",
                          "Items, first-named variable later in alphabet", "Fraction valid "], rows, src)
    hs = d["train_hypothesis_strings"]; po = d["premise_overlap"]
    rows = [["Distinct hypothesis strings in train", hs["n"]], ["...never valid", hs["n_never_valid"]], ["...always valid", hs["n_always_valid"]],
            ["...with both labels", hs["n_both"]],
            ["Share of train items whose hypothesis string is never valid", num(hs["frac_of_train_items_whose_string_is_never_valid"])],
            ["Premises shared by train and dev", po["train&dev"]], ["Premises shared by train and test", po["train&test"]],
            ["Premises shared by dev and test", po["dev&test"]], ["Dev items whose premise is in train", d["n_dev_items_with_premise_in_train"]]]
    txt += "\n" + table(["Quantity (Corr2Cause)", "Value"], rows, src)
    write("benchmark_c2c_description", txt)
    c = r["cladder_full_v1.5"]
    rows = [[names[k], v["n"], v["n_clusters"], ci(v["accuracy"]), ci(v["macro_f1"]), ci(v["auroc"])] for k, v in c["baselines"].items()]
    txt = table(["Text-only baseline (story-held-out cross-validation)", "Items", "Stories", "Accuracy", "Macro-F1", "AUROC"], rows, src)
    rows = [[k, v["n"], num(v["frac_yes"], 2), ci(v["accuracy"]), num(v["auroc"])] for k, v in c["numeric_polarity_by"]["query_type"].items()]
    rows += [[f"rung {k}", v["n"], num(v["frac_yes"], 2), ci(v["accuracy"]), num(v["auroc"])] for k, v in c["numeric_polarity_by"]["rung"].items()]
    txt += "\n" + table(["Query type / rung", "Items", "Fraction yes", "Accuracy of numbers + polarity baseline", "AUROC"], rows, src)
    write("benchmark_cladder", txt)


def t_answer_bias():
    r = load("answer_bias.json")
    if not r:
        return
    rows = []
    for m, tags in r.items():
        for tag, v in tags.items():
            rows.append([LAB[m], DS[tag], v["mode"], v["n"], num(v["yes_rate"], 2), ci(v["accuracy"]), num(v["majority"]),
                         ci(v["auroc"]), ci(v["best_threshold_acc"]), v["reading"]])
    write("answer_bias", table(
        ["Model", "Item set", "Prompt", "Items", "Yes-rate", "Accuracy", "Majority-class accuracy", "AUROC of Yes−No logit difference",
         "Accuracy at best threshold (cross-fitted)", "Reading"], rows, ["answer_bias.json"]))


def t_probe_content():
    rows, rows2, src = [], [], []
    labs = {"correlated": "the two named variables correlate", "adjacent": "they are adjacent in the skeleton (never separated)",
            "separable": "some conditioning set separates them", "name_order": "first-named variable is earlier in the alphabet"}
    for m in MODELS:
        for tag in ("c2c_debiased", "c2c_probe"):
            r = load(f"probe_content_{m}_{tag}.json")
            if not r:
                continue
            src.append(f"probe_content_{m}_{tag}.json")
            for k, v in r["properties"].items():
                rows.append([LAB[m], DS[tag], r["layer"], labs[k], num(v["majority_rate"]), ci(v["probe"]["final"]["accuracy"]),
                             ci(v["probe"]["content_end"]["accuracy"]) if "content_end" in v["probe"] else "–",
                             num(v["gold_accuracy_from_property_alone"])])
            c = r["complementarity"]; k = c["stacked_probe_plus_parsed"]
            rows2.append([LAB[m], DS[tag], ci(c["probe_accuracy"]), ci(c["parsed_accuracy"]), ci(k["accuracy"]),
                          ci(k["minus_parsed_alone_accuracy"], signed=True), num(c["pearson_r_scores"], 2),
                          f"{c['probe_right_when_parsed_wrong']:.2f} (n = {c['n_parsed_wrong']})",
                          ci(c["probe_auroc_where_parsed_says_valid"]), ci(c["probe_auroc_where_parsed_says_invalid"])])
    if rows:
        write("probe_content", table(
            ["Model", "Item set", "Layer", "Property read off the prompt by a parser", "Majority rate", "Probe accuracy, final token",
             "Probe accuracy, end of hypothesis", "Gold-label accuracy from this property alone"], rows, src))
        write("probe_complementarity", table(
            ["Model", "Item set", "Gold probe", "Parsed features", "Probe + parsed features (stacked)", "Stacked − parsed alone",
             "Correlation of the two scores", "Probe right when parsed features are wrong",
             "Probe AUROC where parsed says valid", "Probe AUROC where parsed says invalid"], rows2, src))


if __name__ == "__main__":
    t_behaviour(); t_probes(); t_confound(); t_confound(suffix="_fixedC", name="confound_fixedC"); t_audited(); t_external()
    t_steering(); t_patching(); t_cross_model(); t_benchmarks(); t_answer_bias(); t_probe_content()
