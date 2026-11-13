"""All Tier 1 figures, drawn only from results/*.json. Writes figures/*.png.

Colour follows the entity: each model keeps one hue in every figure; each probe target /
control keeps one hue in every figure. No dual axes; shaded bands are 95% cluster-bootstrap CIs.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from causal_probe.config import FIGURES, MODELS, RESULTS

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODEL_COL = {"qwen0.5b": "#2a78d6", "qwen1.5b": "#eb6834", "gemma2b": "#1baf7a", "qwen0.5b-lora": "#6250d6"}
MODEL_LAB = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B", "gemma2b": "Gemma-2-2B"}
SERIES = {  # probe targets and controls: fixed hue + line style
    "gold": ("#2a78d6", "-", "gold-label probe"),
    "model_out": ("#eb6834", "-", "model-output probe"),
    "control": ("#eda100", "--", "control task (H&L)"),
    "random_init": ("#e87ba4", "--", "random-init model"),
    "shuffled": ("#898781", ":", "shuffled labels"),
}
DATASET_LAB = {"c2c_probe": "Corr2Cause (balanced pool)", "cladder": "CLadder", "c2c_test": "Corr2Cause test"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def probes_file(m, tag):
    """Per-layer probe curves at the C selected (on the selection half) in the confound analysis."""
    c = load(f"confound_{m}_{tag}.json")
    return f"probes_{m}_{tag}_C{c['selected_C_gold']:g}.json" if c else None


def models_with(prefix):
    return [m for m in MODELS if (RESULTS / f"{prefix}_{m}.json").exists()]


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def depth(n):
    return np.arange(n)


# ------------------------------------------------------------------ fig 1: behaviour
def fig_behaviour():
    ms = models_with("behaviour")
    if not ms:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    panels = [("c2c_test", "macro_f1", "Corr2Cause test: macro-F1"),
              ("c2c_test", "accuracy", "Corr2Cause test: accuracy"),
              ("cladder", "accuracy", "CLadder: accuracy")]
    for ax, (tag, metric, title) in zip(axes, panels):
        for i, m in enumerate(ms):
            r = load(f"behaviour_{m}.json")["zero_shot"][tag]
            v = r["logit_scored"][metric]
            ax.bar(i, v["est"], width=0.55, color=MODEL_COL[m])
            ax.errorbar(i, v["est"], yerr=[[v["est"] - v["lo"]], [v["hi"] - v["est"]]], color=INK, lw=1, capsize=3)
            ax.text(i, v["hi"] + 0.015, f"{v['est']:.2f}", ha="center", color=INK2, fontsize=8)
        base = r["baselines"]
        ax.axhline(base["majority"][metric], color=INK2, ls="--", lw=1)
        ax.text(len(ms) - 0.45, base["majority"][metric], " majority", va="bottom", ha="right", color=INK2, fontsize=8)
        ax.axhline(base["random"][metric], color=MUTED, ls=":", lw=1)
        ax.text(len(ms) - 0.45, base["random"][metric], " random", va="top", ha="right", color=MUTED, fontsize=8)
        ax.set_xticks(range(len(ms)), [MODEL_LAB[m] for m in ms])
        ax.set_ylim(0, 1); ax.set_title(title); ax.grid(axis="x", visible=False)
    fig.suptitle("Zero-shot behaviour (answer-token logits). Bars: estimate; whiskers: 95% cluster-bootstrap CI", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig1_behaviour.png")


# ------------------------------------------------------------------ fig 2: logit lens
def fig_lens():
    ms = models_with("lens")
    if not ms:
        return
    tags = ["c2c_probe", "cladder"]
    fig, axes = plt.subplots(len(tags), len(ms), figsize=(3.6 * len(ms), 5.6), squeeze=False, sharey="row")
    for j, m in enumerate(ms):
        res = load(f"lens_{m}.json")
        for i, tag in enumerate(tags):
            ax = axes[i, j]
            # Raw Yes-No lens logit difference by gold label and by whether the final answer is right.
            # (The gold-signed version is a mirror-image artefact when the model's answer is nearly
            # constant: "incorrect" is then just "gold = Yes". Tier 1 audit, finding 5.)
            for key, col, ls in (("gold=yes/correct", "#2a78d6", "-"), ("gold=yes/incorrect", "#2a78d6", "--"),
                                 ("gold=no/correct", "#eb6834", "-"), ("gold=no/incorrect", "#eb6834", "--")):
                c = res[tag]["raw_by_gold_and_correct"][key]
                if c is None or c["n"] < 20:
                    continue
                x = depth(len(c["mean"]))[1:]      # layer 0 (raw embeddings) is off-scale noise under the lens
                g, k = key.split("/")
                ax.plot(x, c["mean"][1:], color=col, ls=ls, lw=1.8, label=f"{g.replace('=', ' ')}, model {k} (n={c['n']})")
                ax.fill_between(x, c["lo"][1:], c["hi"][1:], color=col, alpha=0.12, lw=0)
            ax.axhline(0, color=AXIS, lw=1)
            ax.set_title(f"{MODEL_LAB[m]} · {DATASET_LAB[tag]}", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
            if i == len(tags) - 1:
                ax.set_xlabel("layer (0 = embeddings)")
            if j == 0:
                ax.set_ylabel("lens Yes − No logit difference\n(>0 leans to Yes)")
    fig.suptitle("Logit lens at the final prompt position: colour = gold label, dashes = model's final answer wrong (layer 0 omitted)", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig2_logit_lens.png")

    # 2b: threshold-free view - does the lens logit diff rank gold-Yes items above gold-No items?
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 3.3), sharey=True)
    for ax, tag in zip(axes, tags):
        for m in ms:
            au = load(f"lens_{m}.json")[tag]["auroc_gold_by_layer"]
            ax.plot(np.linspace(0, 1, len(au)), au, color=MODEL_COL[m], label=MODEL_LAB[m])
        ax.axhline(0.5, color=INK2, ls="--", lw=1)
        ax.set_title(DATASET_LAB[tag]); ax.set_xlabel("relative depth (0 = embeddings, 1 = last layer)")
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("AUROC of lens logit diff for the gold label"); axes[0].legend(fontsize=8)
    fig.suptitle("Logit lens: how well does 'what the model would say at this layer' track the gold answer? (dashed = chance)", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig2b_lens_auroc.png")


# ------------------------------------------------------------------ fig 3: probes by layer with controls
def fig_probes():
    for tag in ["c2c_probe", "cladder"]:
        ms = [m for m in MODELS if (RESULTS / f"confound_{m}_{tag}.json").exists()]
        if not ms:
            continue
        fig, axes = plt.subplots(1, len(ms), figsize=(4.1 * len(ms), 3.6), squeeze=False, sharey=True)
        for ax, m in zip(axes[0], ms):
            r = load(probes_file(m, tag))
            for key, (col, ls, lab) in SERIES.items():
                if key not in r["layers"]:
                    continue
                est = np.array([l["accuracy"]["est"] for l in r["layers"][key]])
                x = depth(len(est))
                ax.plot(x, est, color=col, ls=ls, label=lab, lw=2 if ls == "-" else 1.5)
                if key in ("gold", "model_out"):
                    lo = [l["accuracy"]["lo"] for l in r["layers"][key]]
                    hi = [l["accuracy"]["hi"] for l in r["layers"][key]]
                    ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
            c = load(f"confound_{m}_{tag}.json")
            tf = f"tfidf_C{c['selected_C_tfidf']:g}"
            for bname, col, lab in ((tf, INK, "TF-IDF text baseline"), ("type_prior", "#6250d6", "type-prior baseline")):
                ax.axhline(r["baselines"][bname]["accuracy"]["est"], color=col, ls="-.", lw=1, label=lab)
            tb = c.get("text_baselines_all_items", {})
            if "structural" in tb:
                ax.axhline(tb["structural"]["accuracy"], color="#1baf7a", ls="-.", lw=1.4,
                           label="parsed text features + hypothesis prior (post-hoc)")
            if "hyp_prior" in tb:
                ax.axhline(tb["hyp_prior"]["accuracy"], color="#e34948", ls="-.", lw=1.4,
                           label="hypothesis-string prior (post-hoc)")
            ax.axhline(0.5, color=AXIS, lw=1)
            ax.set_title(f"{MODEL_LAB[m]} (model acc. {r['model_accuracy']:.2f}; probe C = {r['probe_C']:g})", fontsize=9)
            ax.set_xlabel("layer (0 = embeddings)")
            ax.set_ylim(0.3, 1.0)
        axes[0, 0].set_ylabel("held-out probe accuracy")
        seen = {}
        for ax in axes[0]:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(ll, hh)
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle(f"Linear probes at the final position · {DATASET_LAB[tag]} · grouped 5-fold × 3 seeds; bands: 95% CI",
                     fontsize=9, color=INK2)
        fig.tight_layout()
        save(fig, f"fig3_probes_{tag}.png")


# ------------------------------------------------------------------ fig 4: selectivity + positions
def fig_selectivity_positions():
    for tag in ["c2c_probe", "cladder"]:
        ms = [m for m in MODELS if (RESULTS / f"confound_{m}_{tag}.json").exists()]
        if not ms:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        for m in ms:
            r = load(probes_file(m, tag))
            sel = np.array(r["selectivity_by_layer"])
            axes[0].plot(np.linspace(0, 1, len(sel)), sel, color=MODEL_COL[m], label=MODEL_LAB[m])
        axes[0].axhline(0, color=AXIS, lw=1)
        axes[0].set_title("Selectivity = gold acc. − control-task acc.")
        axes[0].set_xlabel("relative depth (0 = embeddings, 1 = last layer)"); axes[0].set_ylabel("accuracy points")
        axes[0].legend(fontsize=8)
        m = ms[-1]
        r = load(probes_file(m, tag))
        for key, col, lab in (("gold", "#2a78d6", "final prompt token"),
                              ("gold@content_end", "#1baf7a", "end of hypothesis / question"),
                              ("gold@premise_end", "#eda100", "end of premise / context")):
            est = [l["accuracy"]["est"] for l in r["layers"][key]]
            axes[1].plot(depth(len(est)), est, color=col, label=lab)
        axes[1].axhline(0.5, color=AXIS, lw=1)
        axes[1].set_title(f"Gold probe by token position ({MODEL_LAB[m]})")
        axes[1].set_xlabel("layer (0 = embeddings)"); axes[1].set_ylabel("held-out probe accuracy")
        axes[1].legend(fontsize=8)
        fig.suptitle(DATASET_LAB[tag], fontsize=9, color=INK2)
        fig.tight_layout()
        save(fig, f"fig4_selectivity_positions_{tag}.png")


# ------------------------------------------------------------------ fig 5: the confound
def fig_confound():
    for tag in ["c2c_probe", "cladder"]:
        ms = [m for m in MODELS if (RESULTS / f"confound_{m}_{tag}.json").exists()]
        if not ms:
            continue
        scorers = [("gold_probe", "#2a78d6", "gold probe"),
                   ("structural", "#1baf7a", "parsed text features + hyp. prior (post-hoc)"),
                   ("hyp_prior", "#e34948", "hypothesis-string prior (post-hoc)"),
                   ("tfidf", INK, "TF-IDF"),
                   ("random_init_probe", "#e87ba4", "random-init probe"),
                   ("model_logit_diff", "#eb6834", "model's own logit diff")]
        panels = [("all_items", "accuracy", "Accuracy, all report-half items"),
                  ("model_said_yes", "auroc", "AUROC within 'model leans Yes' (calibrated)"),
                  ("model_said_no", "auroc", "AUROC within 'model leans No' (calibrated)")]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        w = 0.13
        for ax, (sub, metric, title) in zip(axes, panels):
            for i, m in enumerate(ms):
                cj = load(f"confound_{m}_{tag}.json")
                b = cj[sub] if sub == "all_items" else cj["calibrated"][sub]
                if "gold_probe" not in b:
                    ax.text(i, 0.55, f"n={b['n']}\n(too few)", ha="center", color=MUTED, fontsize=8)
                    continue
                for k, (sc, col, lab) in enumerate(scorers):
                    if sc not in b or np.isnan(b[sc][metric]["est"]):
                        continue
                    v = b[sc][metric]
                    x = i + (k - 2.5) * w
                    ax.bar(x, v["est"], width=w * 0.88, color=col, label=lab if i == 0 else None)
                    ax.errorbar(x, v["est"], yerr=[[v["est"] - v["lo"]], [v["hi"] - v["est"]]], color=INK, lw=0.8, capsize=2)
                ax.text(i, 0.965, f"n={b['n']}", ha="center", color=INK2, fontsize=7)
            ax.axhline(0.5, color=INK2, ls="--", lw=1)
            ax.set_xticks(range(len(ms)), [MODEL_LAB[m] for m in ms]); ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, len(ms) - 0.5)
            ax.set_title(title, fontsize=9); ax.grid(axis="x", visible=False)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"Does the gold probe hold more than the text gives away? · {DATASET_LAB[tag]} · report half only; dashed = chance",
                     fontsize=9, color=INK2)
        fig.tight_layout()
        save(fig, f"fig5_confound_{tag}.png")


# ------------------------------------------------------------------ fig 6: steering
DIRS = [("gold_probe", "#2a78d6", "-", "gold-probe direction"), ("gold_diffmeans", "#1baf7a", "-", "gold diff-of-means"),
        ("model_out", "#eb6834", "-", "model-output direction"), ("random_mean", "#898781", "--", "random directions (mean of 3)")]


def fig_steering():
    for tag in ["c2c_probe", "cladder"]:
        ms = [m for m in MODELS if (RESULTS / f"steering_{m}_{tag}.json").exists()]
        if not ms:
            continue
        fig, axes = plt.subplots(2, len(ms), figsize=(3.9 * len(ms), 6), squeeze=False, sharey="row")
        for j, m in enumerate(ms):
            r = load(f"steering_{m}_{tag}.json")
            al = r["alphas"]
            for key, col, ls, lab in DIRS:
                d = r["directions"][key]["add"]
                rnd = key == "random_mean"
                for i, (metric, title) in enumerate((("flip_rate_plus", "answers flipped by +α·direction"),
                                                     ("accuracy_toward_gold", "accuracy when pushed towards the gold answer"))):
                    ax = axes[i, j]
                    est = [x[metric] if rnd else x[metric]["est"] for x in d]
                    ax.plot(al, est, color=col, ls=ls, marker="o", ms=4, label=lab)
                    if not rnd:
                        ax.fill_between(al, [x[metric]["lo"] for x in d], [x[metric]["hi"] for x in d], color=col, alpha=0.15, lw=0)
                    ax.set_xscale("log", base=2); ax.set_ylim(-0.03, 1.03)
                    ax.set_title(f"{MODEL_LAB[m]} · layer {r['layer']}\n{title}", fontsize=8.5)
                    if i == 1:
                        ax.set_xlabel("steering scale α (units of typical activation spread)")
            axes[1, j].axhline(r["baseline"]["accuracy"]["est"], color=INK2, ls=":", lw=1)
        axes[0, 0].set_ylabel("fraction of items"); axes[1, 0].set_ylabel("accuracy")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Steering at the final token · {DATASET_LAB[tag]} · dotted = unsteered accuracy", fontsize=9, color=INK2)
        fig.tight_layout()
        save(fig, f"fig6_steering_{tag}.png")


# ------------------------------------------------------------------ fig 7: patching heatmaps
def fig_patching():
    ms = models_with("patching")
    if not ms:
        return
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])   # two hues + neutral midpoint
    for kind, ktitle in (("hyp", "hypothesis differs by one variable"), ("prem", "premise relabelled (different causal structure)")):
        fig, axes = plt.subplots(2, len(ms), figsize=(4.6 * len(ms), 5.8), squeeze=False, constrained_layout=True)
        for j, m in enumerate(ms):
            r = load(f"patching_{m}.json")
            k = r["kinds"][kind]
            for i, (key, lab) in enumerate((("pooled_recovery", "clean → corrupt"), ("control_pooled_recovery", "random-donor control"))):
                ax = axes[i, j]
                g = np.array(k[key], float).T                    # [regions, layers]
                im = ax.imshow(g, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest")
                ax.set_yticks(range(len(r["regions"])), [x.replace("_", " ") for x in r["regions"]], fontsize=7)
                step = max(1, len(r["layers"]) // 8)
                ax.set_xticks(range(0, len(r["layers"]), step), [r["layers"][t] for t in range(0, len(r["layers"]), step)])
                ax.grid(False)
                ax.set_title(f"{MODEL_LAB[m]} · {lab} (n={k['n_pairs']})", fontsize=8.5)
                if i == 1:
                    ax.set_xlabel("layer patched (0 = embeddings)")
            fig.colorbar(im, ax=axes[:, j], shrink=0.7, label="pooled recovery" if j == len(ms) - 1 else None)
        fig.suptitle(f"Residual-stream patching by layer × token region · minimal pairs: {ktitle}\n"
                     "1 = corrupt run behaves like clean run; grey cells = region empty", fontsize=9, color=INK2)
        save(fig, f"fig7_patching_{kind}.png")


# ------------------------------------------------------------------ fig 8: attribution vs real patching
def fig_components():
    ms = [m for m in models_with("patching") if "validated" in load(f"patching_{m}.json")["kinds"]["prem"].get("components", {})]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.6), squeeze=False)
    for ax, m in zip(axes[0], ms):
        r = load(f"patching_{m}.json")
        for kind, mk in (("prem", "o"), ("hyp", "s")):
            c = r["kinds"][kind].get("components", {})
            if "validated" not in c:
                continue
            for setname, col in (("top", "#2a78d6"), ("random", "#898781")):
                v = [x for x in c["validated"] if x["set"] == setname]
                ax.scatter([x["attribution"] for x in v], [x["real_patching"] for x in v], color=col, marker=mk, s=28,
                           edgecolor=SURF, linewidth=0.8, label=f"{setname} ({kind})")
            for x in sorted(c["validated"], key=lambda x: -abs(x["real_patching"]))[:3]:
                ax.annotate(x["component"], (x["attribution"], x["real_patching"]), fontsize=7, color=INK2,
                            xytext=(3, 3), textcoords="offset points")
        lim = max(abs(np.array(ax.get_xlim())).max(), abs(np.array(ax.get_ylim())).max())
        ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=1)
        c = r["kinds"]["prem"]["components"]
        ax.set_title(f"{MODEL_LAB[m]} (prem: r = {c['pearson_attr_vs_real']:.2f})", fontsize=9)
        ax.set_xlabel("attribution-patching estimate"); ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("real activation patching (pooled recovery)")
    fig.suptitle("Do the cheap gradient estimates agree with real patching? Top-10 and 10 random heads/MLPs; line = perfect agreement",
                 fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig8_attribution_validation.png")


# ------------------------------------------------------------------ fig 9: cross-model
def fig_cross_model():
    r = load("cross_model.json")
    if not r:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for tag, mk in (("c2c_probe", "o"), ("cladder", "s")):
        for m, row in r["models"].items():
            if tag not in row["datasets"]:
                continue
            d = row["datasets"][tag]
            x = row["params_billion"]
            for ax, key in ((axes[0], "gold_probe_accuracy"), (axes[1], "probe_minus_best_text_accuracy")):
                v = d[key]
                ax.errorbar(x, v["est"], yerr=[[v["est"] - v["lo"]], [v["hi"] - v["est"]]], color=MODEL_COL[m], marker=mk,
                            ms=7, capsize=3, mfc=MODEL_COL[m] if tag == "c2c_probe" else SURF, lw=1)
            axes[2].scatter(x, d["selected_layer_relative_depth"], color=MODEL_COL[m], marker=mk, s=50,
                            facecolor=MODEL_COL[m] if tag == "c2c_probe" else SURF)
            axes[2].scatter(x, d["first_layer_within_1pt_relative_depth"], color=MODEL_COL[m], marker=mk, s=14, alpha=0.6)
    axes[0].axhline(0.5, color=INK2, ls="--", lw=1); axes[1].axhline(0, color=INK2, ls="--", lw=1)
    axes[0].set_title("Gold-probe accuracy (report half)"); axes[1].set_title("Probe − strongest text baseline (paired acc.)")
    axes[2].set_title("Relative depth of selected layer\n(small dot: first layer within 1 pt of max)", fontsize=9)
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("parameters (billions)")
        ax.set_xticks([0.5, 1.5, 2.6], ["0.5", "1.5", "2.6"])
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], marker="o", color=INK2, ls="", label="Corr2Cause pool"),
                            Line2D([], [], marker="s", color=INK2, ls="", mfc=SURF, label="CLadder")], fontsize=8)
    fig.suptitle("Does decodability scale with model size? Colour = model; whiskers = 95% cluster-bootstrap CI", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig9_cross_model.png")


if __name__ == "__main__":
    fig_steering(); fig_patching(); fig_components(); fig_cross_model()
    fig_behaviour(); fig_lens(); fig_probes(); fig_selectivity_positions(); fig_confound()
