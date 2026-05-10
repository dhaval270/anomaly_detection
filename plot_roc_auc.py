

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASELINE_PATH  = "results/results_baseline.json"
AUGMENTED_PATH = "results/results_augmented.json"
OUT_DIR        = "results/figures"

CATEGORIES      = ["bottle", "capsule", "carpet"]
MODELS          = ["PaDiM", "PatchCore"]
CORRUPTIONS     = ["gaussian_noise", "gaussian_blur", "brightness_change", "jpeg_compression"]
CORR_LABELS     = ["Gaussian Noise", "Gaussian Blur", "Brightness Change", "JPEG Compression"]
N_SEV           = 5

CAT_COLORS  = {"bottle": "#1f77b4", "capsule": "#ff7f0e", "carpet": "#2ca02c"}
CORR_COLORS = dict(zip(CORRUPTIONS, ["#e41a1c", "#377eb8", "#ff7f00", "#984ea3"]))

os.makedirs(OUT_DIR, exist_ok=True)

with open(BASELINE_PATH)  as f: baseline  = json.load(f)
with open(AUGMENTED_PATH) as f: augmented = json.load(f)


def get_auroc(data, category, model, condition, metric):
    try:
        return data[category][model][condition][metric]
    except KeyError:
        return np.nan


def severity_curve(data, category, model, corruption, metric):
    return [get_auroc(data, category, model, f"{corruption}_sev{s}", metric)
            for s in range(1, N_SEV + 1)]



def plot_auroc_per_category(data, metric, ylabel, filename):
    fig, axes = plt.subplots(
        len(CATEGORIES), len(MODELS),
        figsize=(14, 4 * len(CATEGORIES)),
        sharey=False
    )
    fig.suptitle(f"ROC-AUC per Category — {ylabel} (Baseline)",
                 fontsize=14, fontweight="bold", y=1.01)

    sevs = range(1, N_SEV + 1)
    for r, cat in enumerate(CATEGORIES):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            clean_val = get_auroc(data, cat, model, "clean", metric)

            for corr, label, color in zip(CORRUPTIONS, CORR_LABELS, CORR_COLORS.values()):
                vals = severity_curve(data, cat, model, corr, metric)
                ax.plot(sevs, vals, marker="o", color=color, label=label, linewidth=1.8)

            if not np.isnan(clean_val):
                ax.axhline(clean_val, linestyle="--", color="gray", linewidth=1.2,
                           label=f"Clean ({clean_val:.3f})")

            ax.set_title(f"{cat.capitalize()} — {model}", fontsize=11)
            ax.set_xlabel("Corruption Severity", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(list(sevs))
            ax.set_ylim(0.3, 1.05)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == len(MODELS) - 1:
                ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_auroc_per_category(baseline, "img_auroc", "Image-Level ROC-AUC",
                        "roc_auc_per_category_image.png")
plot_auroc_per_category(baseline, "pix_auroc", "Pixel-Level ROC-AUC",
                        "roc_auc_per_category_pixel.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Baseline vs Augmented AUROC comparison
# Layout: rows = models, cols = corruption types
# Each subplot: one line per category; solid = augmented, dashed = baseline
# ══════════════════════════════════════════════════════════════════════════════
def plot_baseline_vs_augmented(metric, ylabel, filename):
    fig, axes = plt.subplots(
        len(MODELS), len(CORRUPTIONS),
        figsize=(16, 5 * len(MODELS)),
        sharey=False
    )
    fig.suptitle(f"Baseline vs Augmented — {ylabel}",
                 fontsize=14, fontweight="bold", y=1.01)

    sevs = range(1, N_SEV + 1)
    for r, model in enumerate(MODELS):
        for c, (corr, corr_label) in enumerate(zip(CORRUPTIONS, CORR_LABELS)):
            ax = axes[r][c]

            for cat in CATEGORIES:
                color = CAT_COLORS[cat]
                base_vals = severity_curve(baseline,  cat, model, corr, metric)
                aug_vals  = severity_curve(augmented, cat, model, corr, metric)

                ax.plot(sevs, base_vals, marker="o", linestyle="--", color=color,
                        linewidth=1.6, alpha=0.8, label=f"{cat.capitalize()} (Baseline)")
                ax.plot(sevs, aug_vals,  marker="s", linestyle="-",  color=color,
                        linewidth=1.6, alpha=0.8, label=f"{cat.capitalize()} (Augmented)")

                clean_b = get_auroc(baseline, cat, model, "clean", metric)
                ax.axhline(clean_b, linestyle=":", color=color, linewidth=0.8, alpha=0.5)

            ax.set_title(f"{model} — {corr_label}", fontsize=10)
            ax.set_xlabel("Corruption Severity", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(list(sevs))
            ax.set_ylim(0.3, 1.05)
            ax.grid(True, alpha=0.3)

    # unified legend (one entry per category, baseline=dashed, augmented=solid)
    from matplotlib.lines import Line2D
    legend_elements = []
    for cat, color in CAT_COLORS.items():
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle="--", marker="o",
                   label=f"{cat.capitalize()} — Baseline"))
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle="-",  marker="s",
                   label=f"{cat.capitalize()} — Augmented"))

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(CATEGORIES), fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_baseline_vs_augmented("img_auroc", "Image-Level ROC-AUC",
                           "baseline_vs_augmented_image.png")
plot_baseline_vs_augmented("pix_auroc", "Pixel-Level ROC-AUC",
                           "baseline_vs_augmented_pixel.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Clean-condition AUROC bar chart: Baseline vs Augmented per category
# ══════════════════════════════════════════════════════════════════════════════
def plot_clean_bar_comparison(metric, ylabel, filename):
    x = np.arange(len(CATEGORIES))
    width = 0.18
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 5), sharey=True)
    fig.suptitle(f"Clean-Condition ROC-AUC: Baseline vs Augmented — {ylabel}",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODELS):
        base_vals = [get_auroc(baseline,  cat, model, "clean", metric) for cat in CATEGORIES]
        aug_vals  = [get_auroc(augmented, cat, model, "clean", metric) for cat in CATEGORIES]

        bars_b = ax.bar(x - width / 2, base_vals, width, label="Baseline",
                        color="#4e79a7", edgecolor="white")
        bars_a = ax.bar(x + width / 2, aug_vals,  width, label="Augmented",
                        color="#f28e2b", edgecolor="white")

        for bar in list(bars_b) + list(bars_a):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(model, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in CATEGORIES], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0.7, 1.08)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


plot_clean_bar_comparison("img_auroc", "Image-Level ROC-AUC",
                          "clean_bar_baseline_vs_augmented_image.png")
plot_clean_bar_comparison("pix_auroc", "Pixel-Level ROC-AUC",
                          "clean_bar_baseline_vs_augmented_pixel.png")

print("\nAll figures saved to:", OUT_DIR)
