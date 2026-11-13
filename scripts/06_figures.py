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
            for key, col, lab in (("correct", "#2a78d6", "model correct"), ("incorrect", "#e34948", "model incorrect")):
                c = res[tag]["gold_signed"][key]
                if c is None:
                    continue
                x = depth(len(c["mean"]))[1:]      # layer 0 (raw embeddings) is off-scale noise under the lens
                ax.plot(x, c["mean"][1:], color=col, label=f"{lab} (n={c['n']})")
                ax.fill_between(x, c["lo"][1:], c["hi"][1:], color=col, alpha=0.18, lw=0)
            ax.axhline(0, color=AXIS, lw=1)
            ax.set_title(f"{MODEL_LAB[m]} · {DATASET_LAB[tag]}", fontsize=9)
            ax.legend(fontsize=7, loc="upper left")
            if i == len(tags) - 1:
                ax.set_xlabel("layer (0 = embeddings)")
            if j == 0:
                ax.set_ylabel("gold-signed answer logit diff\n(>0 leans to correct answer)")
    fig.suptitle("Logit lens at the final prompt position, split by whether the model's final answer is correct (layer 0 omitted)", fontsize=9, color=INK2)
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
        ms = [m for m in MODELS if (RESULTS / f"probes_{m}_{tag}.json").exists()]
        if not ms:
            continue
        fig, axes = plt.subplots(1, len(ms), figsize=(4.1 * len(ms), 3.6), squeeze=False, sharey=True)
        for ax, m in zip(axes[0], ms):
            r = load(f"probes_{m}_{tag}.json")
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
            for bname, col, lab in (("tfidf", INK, "TF-IDF text baseline"), ("type_prior", "#6250d6", "type-prior baseline")):
                ax.axhline(r["baselines"][bname]["accuracy"]["est"], color=col, ls="-.", lw=1, label=lab)
            ax.axhline(0.5, color=AXIS, lw=1)
            ax.set_title(f"{MODEL_LAB[m]} (model acc. {r['model_accuracy']:.2f})", fontsize=9)
            ax.set_xlabel("layer (0 = embeddings)")
            ax.set_ylim(0.3, 1.0)
        axes[0, 0].set_ylabel("held-out probe accuracy")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle(f"Linear probes at the final position · {DATASET_LAB[tag]} · grouped 5-fold × 3 seeds; bands: 95% CI",
                     fontsize=9, color=INK2)
        fig.tight_layout()
        save(fig, f"fig3_probes_{tag}.png")


# ------------------------------------------------------------------ fig 4: selectivity + positions
def fig_selectivity_positions():
    for tag in ["c2c_probe", "cladder"]:
        ms = [m for m in MODELS if (RESULTS / f"probes_{m}_{tag}.json").exists()]
        if not ms:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        for m in ms:
            r = load(f"probes_{m}_{tag}.json")
            sel = np.array(r["selectivity_by_layer"])
            axes[0].plot(np.linspace(0, 1, len(sel)), sel, color=MODEL_COL[m], label=MODEL_LAB[m])
        axes[0].axhline(0, color=AXIS, lw=1)
        axes[0].set_title("Selectivity = gold acc. − control-task acc.")
        axes[0].set_xlabel("relative depth (0 = embeddings, 1 = last layer)"); axes[0].set_ylabel("accuracy points")
        axes[0].legend(fontsize=8)
        m = ms[-1]
        r = load(f"probes_{m}_{tag}.json")
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
        scorers = [("gold_probe", "#2a78d6", "gold probe"), ("tfidf", INK, "TF-IDF"),
                   ("type_prior", "#6250d6", "type prior"), ("random_init_probe", "#e87ba4", "random-init probe"),
                   ("model_logit_diff", "#eb6834", "model's own logit diff")]
        panels = [("model_wrong", "accuracy", "Accuracy on items the model gets WRONG"),
                  ("model_said_yes", "auroc", "AUROC within 'model said Yes'"),
                  ("model_said_no", "auroc", "AUROC within 'model said No'")]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        w = 0.15
        for ax, (sub, metric, title) in zip(axes, panels):
            for i, m in enumerate(ms):
                b = load(f"confound_{m}_{tag}.json")[sub]
                if "gold_probe" not in b:
                    ax.text(i, 0.55, f"n={b['n']}\n(too few)", ha="center", color=MUTED, fontsize=8)
                    continue
                for k, (sc, col, lab) in enumerate(scorers):
                    if sc not in b or np.isnan(b[sc][metric]["est"]):
                        continue
                    v = b[sc][metric]
                    x = i + (k - 2) * w
                    ax.bar(x, v["est"], width=w * 0.88, color=col, label=lab if i == 0 else None)
                    ax.errorbar(x, v["est"], yerr=[[v["est"] - v["lo"]], [v["hi"] - v["est"]]], color=INK, lw=0.8, capsize=2)
                ax.text(i, 0.965, f"n={b['n']}", ha="center", color=INK2, fontsize=7)
            ax.axhline(0.5, color=INK2, ls="--", lw=1)
            ax.set_xticks(range(len(ms)), [MODEL_LAB[m] for m in ms]); ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, len(ms) - 0.5)
            ax.set_title(title, fontsize=9); ax.grid(axis="x", visible=False)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"Does the gold probe know more than the model says? · {DATASET_LAB[tag]} · report half only; dashed = chance",
                     fontsize=9, color=INK2)
        fig.tight_layout()
        save(fig, f"fig5_confound_{tag}.png")


if __name__ == "__main__":
    fig_behaviour(); fig_lens(); fig_probes(); fig_selectivity_positions(); fig_confound()
