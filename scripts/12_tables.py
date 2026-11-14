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
       "qwen3b": "Qwen2.5-3B", "qwen7b-4bit": "Qwen2.5-7B (4-bit)", "qwen7b": "Qwen2.5-7B"}
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


OTHER_DRAW = {"qwen3b"}     # models whose CLadder results are on the GPU machine's draw of the sample (decision D28)


def ds_(m, tag):
    """Dataset label; CLadder rows on the other draw of the sample carry a dagger."""
    return DS[tag] + (" †" if m in OTHER_DRAW and tag.startswith("cladder") else "")


def table(header, rows, sources):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    dagger = any("†" in str(c) for r in rows for c in r)
    lines += ["", f"<sub>Source: {', '.join(f'`results/{s}`' for s in sources)}</sub>"
              + ("<br><sub>† computed on the GPU machine's draw of the 3,000-item CLadder sample (2,271 items shared with the other models' draw; Section 7).</sub>" if dagger else ""), ""]
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
            rows.append([LAB[m], ds_(m, tag), z["n"], z["n_groups"], ci(s["accuracy"]), ci(s["macro_f1"]), ci(s["auroc_logit_diff"]),
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
            rows.append([LAB[m], ds_(m, tag.removesuffix("_sub_fewshot")), f["n"], ci(f["zero_shot_same_items"]["accuracy"]),
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
            rows.append([LAB[m], ds_(m, tag), f"{L}/{p['n_layers_plus_embed'] - 1}", f"{c['selected_C_gold']:g}",
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
                    rows.append([LAB[m], ds_(m, tag), label, w["n"]] + ["–"] * 10)
                    continue
                v = c["verdict"] if label == "raw" else c["verdict"]["calibrated"]
                rows.append([LAB[m], ds_(m, tag), label, w["n"], num(w.get("frac_gold_yes"), 2),
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
                    rows2.append([LAB[m], ds_(m, tag), label, sname.replace("model_said_", "said "), b["n"], ci(b["gold_probe"]["auroc"]),
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
            rows.append([LAB[m], ds_(m, tag), ci(a["gold_probe"]["accuracy"]), v["best_text_baseline"].replace("_", " "),
                         ci(a["best_text"]["accuracy"]), ci(v["all_items_probe_minus_best_text_accuracy"], signed=True),
                         ci(st.get("model_said_yes"), signed=True), ci(st.get("model_said_no"), signed=True),
                         "**yes**" if v["represents_beyond_best_text_baseline"] else "no",
                         "yes" if c["verdict"]["represents_but_doesnt_say"] else "no",
                         "yes" if v["raw_wrong_items_criterion_applicable"] else f"no ({v['raw_wrong_items_frac_gold_yes']:.0%} gold-Yes)"])
            rows2.append([LAB[m], ds_(m, tag), ci(a["gold_probe"]["accuracy"])] +
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
            rows.append([LAB[m], ds_(m, tag), ci(a["gold_probe"]["accuracy"]), ci(ps["content_end"]["accuracy"]),
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
                rows.append([LAB[m], ds_(m, tag), site, lab, ci(x["slope_per_s_at_alpha_1"], signed=True),
                             num(x["frac_items_with_positive_slope_at_alpha_1"], 2), num(x["alpha_1_in_own_sd"], 1),
                             num(x["slope_per_own_sd_at_alpha_1"], 4, signed=True),
                             num(x["effect_of_natural_class_gap_on_logit_diff"], 4, signed=True),
                             num(x["auroc_gold_report_half"]), num(x["auroc_model_answer_report_half"])])
                rows2.append([LAB[m], ds_(m, tag), site, lab, ci(f4["plus"], 2), ci(f4["minus"], 2),
                              ci(f4["controllability_toward_gold"], 2), ci(f4["controllability_away_from_gold"], 2),
                              num(x["ablation"]["mean_abs_delta_logit_diff"], 4), num(r["ablation_noise_floor_logit_diff"], 4),
                              "yes" if x["ablation"]["above_noise_floor"] else "no"])
        for tag, hyp in (("c2c_probe", ""), ("c2c_debiased", ""), ("cladder", ""), ("c2c_probe", "_hyp"), ("c2c_debiased", "_hyp")):
            c = load(f"steering_controls_{m}_{tag}{hyp}.json")
            if not c:
                continue
            src3.append(f"steering_controls_{m}_{tag}{hyp}.json")
            for d, lab in names:
                x = c["real"][d]; mf = x["matched_control_family"]; fam = c["families"]
                rows3.append([LAB[m], ds_(m, tag), "end of hypothesis" if hyp else "final", c["layer"], lab, ci(x["slope_per_s"], signed=True),
                              f"{100 * x['versus']['isotropic']['percentile_of_abs_slope_per_s']:.0f}",
                              num(fam["isotropic"]["abs_slope_per_s"]["median"]), num(fam["isotropic"]["abs_slope_per_s"]["max"]),
                              mf.replace("_", "-label ", 1).replace("diffmeans", "difference of means"),
                              f"{100 * x['versus'][mf]['percentile_of_abs_slope_per_s']:.0f}",
                              num(fam[mf]["abs_slope_per_s"]["median"]), num(fam[mf]["abs_slope_per_s"]["max"]),
                              num(x["versus"][mf]["ratio_to_median_abs_slope_per_s"], 1),
                              num(x["own_sd_over_s"], 4), num(fam[mf]["own_sd_over_s_median"], 4),
                              f"{100 * x['versus'][mf]['percentile_of_abs_slope_per_own_sd']:.0f}",
                              num(x["versus"][mf]["ratio_to_median_abs_slope_per_own_sd"], 1)])
    write("steering", table(
        ["Model", "Item set", "Layer", "Direction", "Slope: Δ(Yes−No logit) per unit s", "Share of items with positive slope",
         "One unit of s in the direction's own SDs", "Slope per own SD", "Effect of moving across the natural gold-class gap",
         "AUROC for gold label", "AUROC for model's answer"], rows, src))
    write("steering_flips", table(
        ["Model", "Item set", "Layer", "Direction", "Flip rate +4s", "Flip rate −4s", "Controllability towards gold (4s)",
         "Controllability away from gold (4s)", "Ablation mean |ΔLD|", "Noise floor", "Ablation above 3× floor"], rows2, src))
    if rows3:
        write("steering_controls", table(
            ["Model", "Item set", "Token", "Layer", "Direction", "Slope per unit s", "Percentile among 100 isotropic directions (|slope|)",
             "Isotropic median |slope|", "Isotropic max |slope|", "Matched control family (20 directions)",
             "Percentile among matched controls", "Matched median |slope|", "Matched max |slope|", "Ratio to matched median",
             "Natural SD along this direction (units of s)", "Natural SD, matched controls (median)",
             "Percentile among matched controls, per natural SD", "Ratio to matched median, per natural SD"], rows3, src3))


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
            rows.append([LAB[m], row["params_billion"], ds_(m, tag), num(d["model_accuracy"]), ci(d["gold_probe_accuracy"]),
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
            rows.append([LAB[m], ds_(m, tag), v["mode"], v["n"], num(v["yes_rate"], 2), ci(v["accuracy"]), num(v["majority"]),
                         ci(v["auroc"]), ci(v["best_threshold_acc"]), v["reading"]])
    write("answer_bias", table(
        ["Model", "Item set", "Prompt", "Items", "Yes-rate", "Accuracy", "Majority-class accuracy", "AUROC of Yes−No logit difference",
         "Accuracy at best threshold (cross-fitted)", "Reading"], rows, ["answer_bias.json"]))


def t_probe_content():
    """Post-hoc analysis on the debiased pool (05b): baseline ladder, incremental value, property probes, strata, answers."""
    glab = {"A0_single_input": "(a0) single-input control: 0.50 by construction",
            "A_no_binding_interaction": "(a) shallow confound control: interaction, nothing bound to the queried variables",
            "B_binding_text_reference": "(b) structure-aware text reference: binds premise statements to the queried variables"}
    blab = {"hypothesis_string_prior": "Hypothesis-string prior", "bag_of_words_premise_only": "Bag-of-words, premise only",
            "bag_of_words_whole_prompt_linear": "Bag-of-words, whole prompt (linear)", "probe_on_shuffled_labels_null": "Probe trained on shuffled labels (20 shuffles)",
            "template_x_premise_counts": "Per template: premise-level counts", "template_x_premise_bag_of_words": "Per template: bag-of-words of the premise",
            "probe_on_random_init_model": "Probe on a randomly initialised copy of the model",
            "template_x_per_variable_statement_counts": "Per template: statement counts for each queried variable separately",
            "template_x_statements_naming_the_pair": "Per template: statements naming the queried pair (lookup)",
            "per_string_premise_bigrams": "Per hypothesis string: bag of premise 1–2-grams (no parser)",
            "per_string_premise_4grams": "Per hypothesis string: bag of premise 1–4-grams (no parser)",
            "hand_parse_all_features": "Hand-written parse, all features (gradient boosting)",
            "strongest_text_only_parse_plus_4grams": "Hand parse stacked with per-string 4-grams (strongest text-only model here)"}
    plab = {"pair_correlated": "queried pair correlates", "pair_adjacent_in_skeleton": "queried pair adjacent in the skeleton (never separated)",
            "pair_has_common_neighbour": "pair has a common neighbour", "common_neighbour_in_a_separating_set": "a common neighbour lies in a separating set",
            "common_neighbour_in_no_separating_set": "a common neighbour lies in no separating set", "six_variables": "premise has six variables",
            "many_correlation_statements": "many correlation statements (≥ median)",
            "any_marginal_independence_statement": "premise states a marginal independence",
            "many_conditional_independence_statements": "many conditional-independence statements (≥ median)"}
    slab = {"template": "hypothesis template", "pair_adjacent_in_skeleton": "pair adjacent in skeleton", "pair_correlated": "pair correlated",
            "pair_has_common_neighbour": "pair has a common neighbour", "template_x_adjacent_x_common_neighbour": "template × adjacent × common neighbour",
            "hand_parse_prediction": "hand parse's prediction (2 strata)", "hand_parse_score_quintiles": "quintiles of the hand-parse score",
            "parse_quintile_x_4gram_quintile": "hand-parse quintile × per-string 4-gram quintile"}
    tlab = {"parent": "parent: \"X directly causes Y\"", "has_collider": "has_collider", "has_confounder": "has_confounder",
            "non-child descendant": "non-child descendant", "non-parent ancestor": "non-parent ancestor"}
    rA, rB, rC, rD, rE, rF, src = [], [], [], [], [], [], []
    for m in MODELS:
        r = load(f"probe_content_{m}_c2c_debiased.json")
        if not r or "A0_single_input" not in r.get("baselines", {}):
            continue
        src.append(f"probe_content_{m}_c2c_debiased.json")
        b = r["baselines"]
        rA.append([LAB[m], "—", f"**Gold-label probe, final token, layer {r['layer']}**", ci(b["gold_probe"]["accuracy"]), ci(b["gold_probe"]["auroc"]), "—"])
        for g in ("A0_single_input", "A_no_binding_interaction", "B_binding_text_reference"):
            for k, v in b[g].items():
                if "accuracy" in v:
                    rA.append([LAB[m], glab[g], blab[k], ci(v["accuracy"]), ci(v["auroc"]), ci(v["probe_minus_this_accuracy"], signed=True)])
                else:
                    rA.append([LAB[m], glab[g], blab[k], f"{v['mean']:.3f} ± {v['sd']:.3f} (max {v['max']:.3f})", "–", "–"])
        iv = r["incremental_value"]
        for key, lab in (("over_hand_parse", "hand-written parse"), ("over_strongest_text_only", "strongest text-only model (parse + per-string 4-grams)")):
            g = iv[key]
            rB.append([LAB[m], lab, ci(g["base"]["accuracy"]), ci(g["with_probe"]["accuracy"]), ci(g["gain_accuracy"], signed=True),
                       ci(g["gain_auroc"], signed=True), ci(g["gain_accuracy_all_items"], signed=True)])
        for k, v in r["properties"]["by_property"].items():
            x = v["at_gold_probe_layer"]
            rC.append([LAB[m], plab[k], num(v["majority_rate"], 2), ci(x["trained"]["auroc"]),
                       f"{x['shuffled_null_auroc']['mean']:.3f} ± {x['shuffled_null_auroc']['sd']:.3f}",
                       ci(x["random_init_same_layer_auroc"]) if x["random_init_same_layer_auroc"] else "–",
                       num(x["random_init_best_layer_auroc"]) if x["random_init_best_layer_auroc"] is not None else "–",
                       ci(x["text_no_binding_auroc"]), ci(x["text_per_variable_counts_auroc"]),
                       ci(x["trained_at_end_of_premise_auroc"]) if x["trained_at_end_of_premise_auroc"] else "–",
                       num(v["gold_accuracy_from_property_alone"])])
        s_ = r["stratified"]
        for k, v in s_["by_stratifier"].items():
            rD.append([LAB[m], slab[k], v["n_strata"], num(v["gold_accuracy_from_strata_alone"]), ci(s_["overall_probe_auroc"]), ci(v["probe_auroc_within_strata"])])
        an = r["answers"]
        rE.append([LAB[m], "all templates", 2400, ci(an["within_string_auroc_for_gold"]), "–", "–", "–",
                   ci(an["corr_answer_logit_with_probe_score_string_means_removed"], signed=True), "–"])
        for t, v in an["by_template"].items():
            rE.append([LAB[m], tlab.get(t, t), v["n"], ci(v["within_string_auroc_for_gold"]), num(v["within_string_auroc_for_gold_selection_half"]["est"]),
                       num(v["within_string_auroc_for_gold_report_half"]["est"]), ci(v["within_string_auroc_for_adjacency"]),
                       ci(v["corr_answer_logit_with_probe_score"], signed=True), ci(v["probe_within_string_auroc_for_gold"])])
        rb = r["robustness"]
        rF.append([LAB[m], num(rb["three_seed_ensemble_accuracy"]), " / ".join(f"{x:.3f}" for x in rb["single_seed_accuracy"]),
                   f"{rb['shuffled_label_null_accuracy']['mean']:.3f} ± {rb['shuffled_label_null_accuracy']['sd']:.3f}",
                   ci(rb["trained_on_selection_half_only_accuracy"])])
    if not rA:
        return
    write("debiased_ladder", table(["Model", "Group", "Scorer", "Accuracy", "AUROC", "Gold probe − this (accuracy, paired)"], rA, src))
    write("debiased_incremental", table(
        ["Model", "Probe added to", "Base accuracy", "Base + probe", "Gain in accuracy (report half)", "Gain in AUROC (report half)",
         "Gain in accuracy (all items)"], rB, src))
    write("debiased_properties", table(
        ["Model", "Property of the item", "Majority rate", "Probe AUROC, trained model", "Shuffled-label null (mean ± SD, 10 shuffles)",
         "Random-init model, same layer", "Random-init, its best layer", "Text, no binding (per template: premise counts + string id)",
         "Text, per-variable statement counts", "Trained model at end of premise (hypothesis not yet read)",
         "Gold accuracy from this property alone"], rC, src))
    write("debiased_strata", table(
        ["Model", "Stratified by", "Strata", "Gold accuracy from the strata alone", "Gold-probe AUROC, unstratified",
         "Gold-probe AUROC within strata (pair-weighted)"], rD, src))
    write("debiased_answers", table(
        ["Model", "Hypothesis template", "Items", "Answer logit: AUROC for gold within hypothesis string", "…selection half", "…report half",
         "Answer logit: AUROC for pair adjacency", "Correlation of answer logit with probe score (string means removed)",
         "Gold probe: AUROC within hypothesis string"], rE, src))
    write("debiased_robustness", table(
        ["Model", "Probe accuracy, 3-seed ensemble (report half)", "Single seeds", "Shuffled-label null (20 shuffles)",
         "Probe trained on the selection half only"], rF, src))


def t_patching_same():
    """Label-changing vs label-preserving minimal pairs (round 2)."""
    import numpy as np
    rows, src = [], []
    for m in MODELS:
        ch, sm = load(f"patching_{m}.json"), load(f"patching_{m}_same.json")
        if not sm:
            continue
        src += [f"patching_{m}_same.json"] + ([f"patching_{m}.json"] if ch else [])
        for kind in ("hyp", "prem"):
            for lab, r in (("label changes (Yes → No)", ch), ("label preserved", sm)):
                if not r:
                    continue
                k = r["kinds"][kind]; b = k["behaviour_on_pairs"]
                fin = np.array(k["pooled_recovery"], float)[:, -1]
                first = next((r["layers"][i] for i, v in enumerate(fin) if v >= 0.5), None)
                byl = k.get("by_common_label", {})
                rows.append([LAB[m], kind, lab, k["n_pairs"],
                             " / ".join(str(byl[c]["n_pairs"]) for c in ("1", "0") if c in byl) or "–",
                             ci(b["mean_delta"], signed=True), num(b["mean_abs_delta"]),
                             " / ".join(num(byl[c]["mean_abs_delta"]) for c in ("1", "0") if c in byl) or "–",
                             first if first is not None else "–", f"{first / r['layers'][-1]:.2f}" if first is not None else "–"])
    if rows:
        write("patching_label_preserving", table(
            ["Model", "Pair kind", "Pairs", "n", "of which Yes/Yes, No/No", "Mean D = LD(original) − LD(edited)", "Mean |D|",
             "Mean |D|: Yes/Yes, No/No", "First layer with ≥ 0.5 recovery at final token", "…as relative depth"], rows, src))


FEATURE_SITES = (("adj_final_best", "final token, best layer"), ("adj_hyp_best", "end of hypothesis, best layer"),
                 ("adj_final_first90", "final token, first layer at 90%"), ("adj_hyp_first90", "end of hypothesis, first layer at 90%"))
TSHORT = {"parent": "parent", "non-parent ancestor": "ancestor", "non-child descendant": "descendant",
          "has_collider": "collider", "has_confounder": "confounder"}


def t_feature_causal():
    """Interventions on the adjacency direction (round 2, scripts/16_feature_causal.py)."""
    dn = (("feat_probe", "adjacency probe weights"), ("feat_dom", "adjacency difference of means"),
          ("gold_probe", "gold-label probe weights"), ("model_out", "model-answer probe weights"))
    r1, r2, r3, r4, src = [], [], [], [], []
    for m in MODELS:
        for site, slab in FEATURE_SITES:
            r = load(f"feature_causal_{m}_c2c_debiased_{site}.json")
            if not r:
                continue
            src.append(f"feature_causal_{m}_c2c_debiased_{site}.json")
            tms = sorted(r["observational_by_template"], key=lambda k: list(TSHORT).index(k))
            where = f"{slab} ({r['layer']}/{r['n_layers']})"
            for d, lab in dn:
                x = r["slope_per_own_sd"][d]; pt = x["pattern_across_templates"]
                r1.append([LAB[m], where, r["n_items"], lab, ci(x["pooled"], 4, signed=True),
                           f"{100 * x['pooled']['vs_matched_controls']['percentile']:.0f}",
                           num(x["pooled"]["vs_matched_controls"]["control_max"], 4, signed=True),
                           " · ".join(f"{TSHORT[k]} {x['by_template'][k]['est']:+.3f}" for k in tms),
                           f"{pt['n_sign_agreements']}/{len(tms)}", num(pt["pearson_r"]["value"], 2, signed=True),
                           f"{100 * pt['pearson_r']['percentile']:.0f}",
                           num(pt["range_across_templates"]["value"], 3), f"{100 * pt['range_across_templates']['percentile']:.0f}"])
            for k in tms:
                o = r["observational_by_template"][k]
                for d, lab in dn[:2]:
                    x = r["slope_per_own_sd"][d]["by_template"][k]
                    sh = r["shift_to_other_class"][d]["by_template"][k]
                    r2.append([LAB[m], where, TSHORT[k], lab, num(o["within_string_auroc_answer_for_property"]),
                               num(o["mean_logit_diff_gap_property_1_minus_0"], 3, signed=True), ci(x, 4, signed=True),
                               f"{100 * x['signed_percentile_among_controls']:.0f}", ci(sh, 3, signed=True),
                               (f"{100 * sh['est'] / o['mean_logit_diff_gap_property_1_minus_0']:+.0f}%"
                                if abs(o["mean_logit_diff_gap_property_1_minus_0"]) > 1e-6 else "–"),
                               f"{100 * sh['vs_matched_controls']['percentile']:.0f}",
                               num(sh["auroc_property_baseline"]), num(sh["auroc_property_after_shift"]),
                               num(sh["auroc_property_after_shift_controls_mean"])])
            for d, x in r["ablation"].items():
                lab = dict(dn).get(d, f"rank-{r['rank']} adjacency subspace")
                for k in tms:
                    y = x["by_template"][k]
                    r3.append([LAB[m], where, lab, TSHORT[k], num(y["baseline"]["auroc_gold"]),
                               num(y["change_in_auroc_gold"]["value"], 3, signed=True),
                               f"{num(y['change_in_auroc_gold']['control_min'], 3, signed=True)} to {num(y['change_in_auroc_gold']['control_max'], 3, signed=True)}",
                               num(y["baseline"]["auroc_property"]),
                               num(y["change_in_auroc_property"]["value"], 3, signed=True),
                               f"{num(y['change_in_auroc_property']['control_min'], 3, signed=True)} to {num(y['change_in_auroc_property']['control_max'], 3, signed=True)}",
                               num(y["mean_abs_delta_logit_diff"]["value"], 3), num(y["mean_abs_delta_logit_diff"]["control_max"], 3)])
            g = r["direction_geometry"]
            r4.append([LAB[m], where, num(r["selection_auroc_at_layer"]), num(g["report_half_auroc_for_property"]["feat_probe"]),
                       num(g["report_half_auroc_for_property"]["feat_dom"]), num(r["class_gap_in_own_sd"]["feat_probe"], 2),
                       num(r["class_gap_in_own_sd"]["feat_dom"], 2), num(g["cos_feat_probe_vs_feat_dom"], 2),
                       num(g["cos_feat_probe_vs_gold_probe"], 2), num(g["cos_feat_probe_vs_model_out"], 2),
                       num(r["baseline_logit_diff_sd"], 2), num(r["baseline_max_abs_diff_vs_cache"], 2)])
    if not src:
        return
    write("v1_adjacency_sites", table(
        ["Model", "Site (layer)", "Adjacency AUROC, selection half (CV)", "…report half, probe direction", "…report half, difference of means",
         "Class gap along probe direction (own SDs)", "…along difference of means", "cos(probe, diff. of means)",
         "cos(adjacency probe, gold probe)", "cos(adjacency probe, model-answer probe)", "SD of the answer logit difference",
         "Max |unsteered − cached| logit difference"], r4, src))
    write("v1_adjacency_slopes", table(
        ["Model", "Site (layer)", "Items", "Direction", "Slope: Δ(Yes−No logit) per natural SD, all templates",
         "Percentile of |slope| among 20 matched controls", "Largest matched-control slope", "Slope by hypothesis template",
         "Signs agreeing with the observational pattern", "Correlation with the observational pattern (5 templates)",
         "Percentile of that correlation among controls", "Range of slopes across templates", "Percentile of that range among controls"], r1, src))
    write("v1_adjacency_by_template", table(
        ["Model", "Site (layer)", "Template", "Direction", "Observed: AUROC of answer for adjacency (within string)",
         "Observed: logit-difference gap, adjacent − not adjacent", "Steering slope per natural SD", "Signed percentile among matched controls",
         "Shift to the other class: effect towards Yes per push towards adjacent", "…as a share of the observed gap", "Percentile among matched controls",
         "AUROC of answer for adjacency: unsteered", "…after the shift", "…after the shift, control mean"], r2, src))
    write("v1_adjacency_ablation", table(
        ["Model", "Site (layer)", "Removed", "Template", "AUROC of answer for gold label: unsteered", "Change after removal",
         "Change, range over 20 controls", "AUROC of answer for adjacency: unsteered", "Change after removal", "Change, range over 20 controls",
         "Mean |Δ logit difference|", "…largest control"], r3, src))


TEMPLATES_T = ["parent", "non-parent ancestor", "non-child descendant", "has_collider", "has_confounder"]
ADJ_SITES = (("final_best", "final token, best layer"), ("hyp_best", "end of hypothesis, best layer"),
             ("final_first50", "final token, earlier layer"), ("hyp_first50", "end of hypothesis, earlier layer"))


def rng_(c, d=3):
    return f"{num(c['control_min'], d, signed=True)} to {num(c['control_max'], d, signed=True)}"


def t_adjacency():
    """The corrected adjacency test (round 2, scripts/17_adjacency_test.py), with the readouts added after the fifth audit."""
    r1, r2, r3, src = [], [], [], []
    for m in MODELS:
        for site, slab in ADJ_SITES:
            r = load(f"adjacency_{m}_c2c_debiased_{site}.json")
            if not r or "audit5" not in r:
                continue
            src.append(f"adjacency_{m}_c2c_debiased_{site}.json")
            where = f"{slab} ({r['layer']}/{r['n_layers']})"
            a5 = r["audit5"]; q = a5["erasure_quality_on_held_out_items"]
            lv = r["leace"]["adjacency_auroc_of_fresh_probe_report_half"]
            cv, ev = q["adjacency_auroc_cv_within_report_half"], q["adjacency_auroc_trained_on_unevaluated_report_items_tested_on_evaluated"]
            r1.append([LAB[m], where, r["n_items"], num(a5["within_string_answer_linearly_predictable"]["held_out_r2"], 2),
                       num(cv["before"]), num(cv["after_erasure"]), num(ev["before"]), num(ev["after_erasure"]),
                       num(lv["after_erasure"]["pooled"]), num(lv["after_removing_probe_direction_only"]["pooled"]),
                       num(a5["erasure_is_translation"]["pearson_r_observed_vs_minus_coefficient_times_translation"], 3),
                       num(r["self_check_shortcut_vs_full_forward_max_abs_diff"], 3), num(r["baseline_logit_diff_sd"], 2)])
            gb = a5["gaps_on_evaluated_items_within_string"]["by_template"]
            for g in r["observational"]:
                o = r["observational"][g]; tr = r["translation"]["by_template"][g]
                pg = r["probe_direction_across_class_gap"]["by_template"][g]; e = gb[g]
                fc = e["fair_controls_change_in_absolute_gap"]
                r2.append([LAB[m], where, TSHORT[g], ci(o["gap_adjacent_minus_not"], signed=True),
                           ci(tr, signed=True), rng_(tr["vs_controls_norm_matched"]),
                           f"{100 * tr['vs_controls_norm_matched']['share_of_controls_smaller_in_absolute_value']:.0f}",
                           num(e["gap_before"], 3, signed=True), num(e["gap_after_erasure"], 3, signed=True),
                           num(e["change_in_absolute_gap"], 3, signed=True), rng_(fc),
                           f"{100 * fc['share_of_controls_smaller_in_absolute_value']:.0f}", ci(pg, 4, signed=True)])
            tc = r["translation"]["contrast"]; v = a5["translation_contrast_under_other_sign_choices"]
            un = v["uniform_sign_adjacent_towards_no"]; mg = a5["gaps_on_evaluated_items_within_string"]["mean_absolute_gap_all_five_templates"]
            mo = r["steps"]["model_out"]["per_unit_norm_from_large_step"]
            r3.append([LAB[m], where, ", ".join(TSHORT[g] for g in r["templates_in_contrast"]) or "none", num(r["full_mediation_benchmark"]),
                       num(tc["est"], 3, signed=True), rng_(tc["vs_controls_norm_matched"]),
                       f"{100 * tc['vs_controls_norm_matched']['share_of_controls_smaller_in_absolute_value']:.0f}",
                       num(un["value"], 3, signed=True), f"{100 * un['share_of_controls_smaller_in_absolute_value']:.0f}",
                       f"{a5['n_translation_effects_towards_no']}/5",
                       f"{num(mg['before'])} → {num(mg['after_erasure'])}", ci(mg["reduction"], signed=True),
                       f"{100 * mg['reduction']['vs_fair_controls']['share_of_controls_smaller_in_absolute_value']:.0f}",
                       f"{mo['est']:+.3f} ({100 * mo['vs_controls']['share_of_controls_smaller_in_absolute_value']:.0f}%)"])
    r4, r5, src4 = [], [], []
    for m in MODELS:
        for site, slab in ADJ_SITES:
            r = load(f"adjacency_{m}_c2c_debiased_{site}.json")
            if not r or "transfer" not in r:
                continue
            src4.append(f"adjacency_{m}_c2c_debiased_{site}.json")
            where = f"{slab} ({r['layer']}/{r['n_layers']})"
            x = r["transfer"]; T = x["templates"]; M = x["effect_matrix_rows_target_cols_source"]; C = x["cosine_between_class_mean_differences"]
            for gi, g in enumerate(T):
                r4.append([LAB[m], where, TSHORT[g], num(r["observational"][g]["gap_adjacent_minus_not"]["est"], 3, signed=True)]
                          + [num(M[gi][hi], 3, signed=True) for hi in range(len(T))])
            off = [C[i][j] for i in range(len(T)) for j in range(len(T)) if i < j]
            tg, sr = x["off_diagonal_contrast_with_the_TARGET_template_sign"], x["off_diagonal_contrast_with_the_SOURCE_template_sign"]
            r5.append([LAB[m], where, ci(tg, signed=True), ci(sr, signed=True), num(r["full_mediation_benchmark"]),
                       f"{min(off):+.2f} to {max(off):+.2f}",
                       " · ".join(f"{TSHORT[T[j]]} {C[0][j]:+.2f}" for j in range(1, len(T)))])
    if src4:
        write("adjacency_transfer", table(
            ["Model", "Site (layer)", "Item's template (target)", "Target's observed gap"]
            + [f"Vector from: {TSHORT[g]}" for g in TEMPLATES_T], r4, src4))
        write("adjacency_transfer_summary", table(
            ["Model", "Site (layer)", "Off-diagonal contrast with the target template's sign", "…with the source template's sign",
             "Full-mediation benchmark", "Cosines between the five class-mean differences (range)",
             "Cosine of the direct-causation difference with each of the others"], r5, src4))
    if not src:
        return
    write("adjacency_sites", table(
        ["Model", "Site (layer)", "Items", "Within-question answer linearly predictable from this site (held-out R²)",
         "Adjacency AUROC on held-out items: before", "…after erasure", "Adjacency AUROC on the evaluated items: before", "…after erasure",
         "(Probe trained on the half the eraser was fitted on: after erasure)", "(…after removing the probe direction only)",
         "Erasure effect vs −coefficient × translation effect (r over items)", "Self-check: shortcut vs full forward pass (logits)",
         "SD of the answer logit difference"], r1, src))
    write("adjacency_effects", table(
        ["Model", "Site (layer)", "Template", "Observed gap in the answer logit, adjacent − not adjacent (all items)",
         "Translation by the class-mean difference: effect towards Yes per move towards adjacent", "…20 random directions of the same size (range)",
         "Controls smaller in absolute value (%)", "Gap on evaluated items: before", "…after erasure", "Change in absolute gap",
         "…20 fair controls: same per-item coefficients, random direction (range)", "Controls smaller in absolute value (%)",
         "Moving across the class gap along the probe's weight vector"], r2, src))
    write("adjacency_contrast", table(
        ["Model", "Site (layer)", "Templates in the pre-specified contrast", "Full-mediation benchmark (mean |observed gap|)",
         "Translation: pre-specified contrast", "…20 random directions of the same size (range)", "Controls smaller in absolute value (%)",
         "Same with one sign for all types (adjacent → No)", "Controls smaller (%)", "Translation effects towards No",
         "Mean absolute gap on evaluated items, before → after erasure", "Reduction (interval over items)", "Fair controls smaller in absolute value (%)",
         "Positive control: model-answer direction, slope per unit norm (share of controls it exceeds)"], r3, src))


def t_sae():
    """Tier 3: SAE features for causal structure on Gemma-2-2B (first Kaggle run)."""
    f, a = load("sae_features.json"), load("sae_ablation.json")
    if not f:
        return
    r1, r2, r3 = [], [], []
    for layer, lv in f["layers"].items():
        for pos, pv in lv["positions"].items():
            rc = pv["reconstruction"]
            r1.append([layer, pos.replace("_", " "), num(rc["fraction_variance_unexplained"], 2), num(rc["relative_squared_error"], 2),
                       num(rc["cosine_recon"], 2), num(rc["mean_l0"], 0), pv["n_features_ever_active"]])
            for st, sv in pv["structures"].items():
                top = sv["top_features"]
                best = max(top, key=lambda t: t["auroc_heldout_stories"])
                import numpy as np
                r2.append([layer, pos.replace("_", " "), st, sv["n_target_eval"], num(float(np.mean([t["auroc_discovery"] for t in top])), 2),
                           num(float(np.mean([t["auroc_heldout_stories"] for t in top])), 2),
                           f"{best['feature']} ({best['auroc_heldout_stories']:.2f})",
                           sum(t["auroc_heldout_stories"] >= 0.7 for t in top)])
    write("sae_reconstruction", table(
        ["Block", "Token", "Fraction of variance unexplained", "Relative squared error", "Cosine(reconstruction, activation)",
         "Mean active features", "Features ever active"], r1, ["sae_features.json"]))
    write("sae_features", table(
        ["Block", "Token", "Graph type", "Held-out items of that type", "Mean AUROC of the 10 selected features, discovery stories",
         "…held-out stories", "Best feature (held-out AUROC)", "Features with held-out AUROC ≥ 0.70"], r2, ["sae_features.json"]))
    if a:
        for st, sv in a["structures"].items():
            for where, key in (("items of that graph type", "target_structure"), ("items of other graph types", "other_structures")):
                x = sv[key]
                r3.append([st, where, x["n"], ci(x["mean_abs_delta_top"]), num(x["mean_abs_delta_random"]),
                           " / ".join(num(v, 2) for v in x["mean_abs_delta_random_each"]),
                           # the stored "top_minus_random" interval subtracts |mean over random sets|, not the mean of
                           # their absolute effects, so it overstates the gap; the difference of the two columns is shown
                           num(x["mean_abs_delta_top"]["est"] - x["mean_abs_delta_random"], 3, signed=True),
                           num(x["mean_signed_delta_top"], 3, signed=True), num(x["flip_rate_top"], 2), num(x["flip_rate_random"], 2),
                           num(x["accuracy_baseline"], 2), num(x["accuracy_top_ablated"], 2)])
        write("sae_ablation", table(
            ["Graph type whose top-10 features are removed", "Evaluated on", "Items", "Mean |Δ logit difference|, top features",
             "…random features (mean of 3 sets)", "…each random set", "Top − random (difference of the two means)", "Mean signed Δ", "Flip rate, top", "Flip rate, random",
             "Accuracy before", "Accuracy after"], r3, ["sae_ablation.json"]))


if __name__ == "__main__":
    t_behaviour(); t_probes(); t_confound(); t_confound(suffix="_fixedC", name="confound_fixedC"); t_audited(); t_external()
    t_steering(); t_patching(); t_cross_model(); t_benchmarks(); t_answer_bias(); t_probe_content()
    t_patching_same(); t_adjacency(); t_sae()      # t_feature_causal(): first version of the adjacency test, superseded
