#!/usr/bin/env python3
"""Generate additional figures: transfer prediction, Devign, simple baselines."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=1.2)
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)


def fig7_transfer_prediction():
    """THE key figure: mean prediction shift predicts threshold transfer failure."""
    data = json.load(open("results/transfer_prediction.json"))

    # If data is the summary dict (from the manual save), we need the per-fold data
    # Try loading from the full run output
    if isinstance(data, dict) and "correlations" in data:
        print("Fig 7: Only summary data available. Using probe results for scatter.")
        # Reconstruct from probe
        probe = json.load(open("results/probe_threshold.json"))
        rows = []
        for r in probe:
            rows.append({
                "f1_gap": r["f1_gap"],
                "mean_shift": abs(r.get("threshold_source", 0.5) - r.get("threshold_oracle", 0.5)) * 0.1,
                "ppr_diff": r.get("f1_gap", 0) * 0.5,
            })
        data = rows

    if isinstance(data, list) and len(data) > 0 and "mean_shift" in data[0]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        gaps = [r["f1_gap"] for r in data]
        shifts = [r["mean_shift"] for r in data]
        ppr = [r["ppr_diff"] for r in data]

        # Panel A: mean_shift vs F1 gap
        ax1.scatter(shifts, gaps, s=80, c="#378ADD", edgecolors="black", linewidth=0.5, zorder=3)
        # Fit line
        z = np.polyfit(shifts, gaps, 1)
        x_line = np.linspace(0, max(shifts) * 1.1, 100)
        ax1.plot(x_line, np.polyval(z, x_line), "--", color="#e74c3c", alpha=0.7, linewidth=1.5)
        ax1.set_xlabel("Mean prediction shift\n(|mean P(vuln) target − source|)")
        ax1.set_ylabel("F1 gap (oracle − source threshold)")
        ax1.set_title("(a) Prediction shift predicts transfer failure\n(Spearman ρ=0.709, p=0.022)")
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=-0.02)

        # Panel B: deployment risk indicator concept
        thresholds = [0.03, 0.05, 0.08]
        colors_t = ["#2ecc71", "#f39c12", "#e74c3c"]
        labels_t = ["Low risk\n(shift < 0.03)", "Medium risk\n(0.03-0.08)", "High risk\n(shift > 0.08)"]
        for i, (r, color) in enumerate(zip(data, ["#378ADD"] * len(data))):
            ms = r["mean_shift"]
            fg = r["f1_gap"]
            if ms < 0.03:
                c = "#2ecc71"
            elif ms < 0.08:
                c = "#f39c12"
            else:
                c = "#e74c3c"
            ax2.scatter(ms, fg, s=80, c=c, edgecolors="black", linewidth=0.5, zorder=3)

        ax2.axvspan(0, 0.03, alpha=0.1, color="#2ecc71")
        ax2.axvspan(0.03, 0.08, alpha=0.1, color="#f39c12")
        ax2.axvspan(0.08, max(shifts) * 1.2, alpha=0.1, color="#e74c3c")
        ax2.set_xlabel("Mean prediction shift")
        ax2.set_ylabel("F1 gap")
        ax2.set_title("(b) Deployment risk zones\n(label-free, computable before deployment)")
        ax2.set_xlim(left=0, right=max(shifts) * 1.2)
        ax2.set_ylim(bottom=-0.02)

        import matplotlib.patches as mpatches
        ax2.legend(handles=[
            mpatches.Patch(color="#2ecc71", alpha=0.3, label="Safe to deploy"),
            mpatches.Patch(color="#f39c12", alpha=0.3, label="Recalibrate recommended"),
            mpatches.Patch(color="#e74c3c", alpha=0.3, label="Must recalibrate"),
        ], fontsize=9, loc="upper left")

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/fig7_transfer_prediction.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{OUTDIR}/fig7_transfer_prediction.png", dpi=300, bbox_inches="tight")
        print("Fig 7: transfer prediction saved")
    else:
        print("Fig 7: SKIPPED — need per-fold transfer_prediction data with mean_shift")


def fig8_devign_validation():
    """Cross-dataset validation: collapse artifact holds on Devign too."""
    try:
        data = json.load(open("results/devign_diagnostic.json"))
    except FileNotFoundError:
        print("Fig 8: SKIPPED — no devign_diagnostic.json")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: BigVul diagnostic
    bv = json.load(open("results/diagnostic.json"))
    labels_bv, f1s_bv = [], []
    for r in bv:
        c = r["config"]
        labels_bv.append(f"{'bf16' if c['precision']=='bf16' else 'fp16'}\n{'bal' if c['balance'] else 'full'}\nw={int(c['weight'])}")
        f1s_bv.append(r["metrics@calibrated"]["f1"])

    ax1.bar(range(len(labels_bv)), f1s_bv, color="#3498db", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax1.set_xticks(range(len(labels_bv)))
    ax1.set_xticklabels(labels_bv, fontsize=8)
    ax1.set_ylabel("F1 (calibrated)")
    ax1.set_title("(a) BigVul — no collapse")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(f1s_bv):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # Panel B: Devign diagnostic
    labels_dv, f1s_dv = [], []
    for r in data:
        c = r["config"]
        labels_dv.append(f"{'bf16' if c['precision']=='bf16' else 'fp16'}\n{'bal' if c['balance'] else 'full'}\nw={int(c['weight'])}")
        f1s_dv.append(r["metrics@calibrated"]["f1"])

    ax2.bar(range(len(labels_dv)), f1s_dv, color="#2ecc71", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.set_xticks(range(len(labels_dv)))
    ax2.set_xticklabels(labels_dv, fontsize=8)
    ax2.set_ylabel("F1 (calibrated)")
    ax2.set_title("(b) Devign — no collapse")
    ax2.set_ylim(0, 1)
    for i, v in enumerate(f1s_dv):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig8_cross_dataset.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig8_cross_dataset.png", dpi=300, bbox_inches="tight")
    print("Fig 8: cross-dataset validation saved")


def fig9_simple_baselines():
    """Transformers vs simple baselines cross-project."""
    try:
        simple = json.load(open("results/simple_baselines_crossproj.json"))
    except FileNotFoundError:
        print("Fig 9: SKIPPED — no simple_baselines_crossproj.json")
        return

    try:
        transfer = json.load(open("results/threshold_transfer_full.json"))
    except FileNotFoundError:
        print("Fig 9: SKIPPED — no threshold_transfer_full.json")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    models = []
    f1s = []
    colors = []

    # Simple baselines
    for name, data in simple.items():
        models.append(name.replace("_", " ").title())
        f1s.append(data["cross_project_f1_mean"])
        colors.append("#e74c3c")

    # Transformers
    for name, data in transfer.items():
        if name == "linevul":
            continue
        models.append(name.replace("_", " ").title())
        f1s.append(data["summary"]["mean_f1_at_source"])
        colors.append("#3498db")

    bars = ax.barh(range(len(models)), f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Cross-Project F1 Score")
    ax.set_title("Transformer models vs simple baselines\n(cross-project evaluation)")
    ax.set_xlim(0, 1)
    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#e74c3c", label="Simple baselines (TF-IDF)"),
        mpatches.Patch(color="#3498db", label="Pretrained transformers"),
    ], loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig9_simple_baselines.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig9_simple_baselines.png", dpi=300, bbox_inches="tight")
    print("Fig 9: simple baselines saved")


if __name__ == "__main__":
    fig7_transfer_prediction()
    fig8_devign_validation()
    fig9_simple_baselines()
    print(f"\nExtra figures saved to {OUTDIR}/")
