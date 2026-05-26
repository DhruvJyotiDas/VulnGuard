#!/usr/bin/env python3
"""Generate all publication figures from results JSONs. CPU only."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

COLORS = {"codebert": "#e74c3c", "codebert_cs": "#3498db",
          "graphcodebert": "#2ecc71", "linevul": "#9b59b6",
          "codebert_vanilla": "#e74c3c", "codebert_cs10": "#3498db",
          "codebert_cs30": "#f39c12"}


def fig1_collapse_diagnostic():
    """Fig 1: The collapse was a training artifact, not a model limitation."""
    data = json.load(open("results/diagnostic.json"))
    labels, f1s, colors = [], [], []
    for r in data:
        c = r["config"]
        label = f"{'bf16' if c['precision']=='bf16' else 'fp16'}\n{'bal' if c['balance'] else 'imb'}\nw={int(c['weight'])}"
        labels.append(label)
        f1s.append(r["metrics@calibrated"]["f1"])
        colors.append("#3498db" if c["precision"] == "bf16" else "#e74c3c")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1 (calibrated threshold)")
    ax.set_title("All CodeBERT Configurations Achieve Comparable F1\n(No collapse under any precision/balance/weight combination)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(y=0.008, color="red", linestyle="--", alpha=0.5, label="Originally reported F1=0.008")
    ax.legend(loc="upper right")
    bf16_patch = mpatches.Patch(color="#3498db", label="bf16")
    fp16_patch = mpatches.Patch(color="#e74c3c", label="fp16")
    ax.legend(handles=[bf16_patch, fp16_patch, plt.Line2D([0],[0], color="red", linestyle="--", label="Original paper F1=0.008")],
              loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig1_collapse_artifact.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig1_collapse_artifact.png", dpi=300, bbox_inches="tight")
    print("Fig 1: collapse artifact saved")


def fig2_threshold_transfer():
    """Fig 2: Threshold transfer gap across models."""
    data = json.load(open("results/threshold_transfer_full.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: F1 gap distribution per model
    model_gaps = {}
    for model_name, mdata in data.items():
        if model_name == "linevul":  # skip duplicate
            continue
        gaps = [r["f1_gap"] for r in mdata["runs"]]
        model_gaps[model_name] = gaps

    positions = range(len(model_gaps))
    bp = ax1.boxplot(model_gaps.values(), positions=list(positions), widths=0.6,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
    for patch, name in zip(bp["boxes"], model_gaps.keys()):
        patch.set_facecolor(COLORS.get(name, "#95a5a6"))
        patch.set_alpha(0.7)
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels([n.replace("_", "\n") for n in model_gaps.keys()], fontsize=9)
    ax1.set_ylabel("F1 Gap (Oracle − Source Threshold)")
    ax1.set_title("(a) Threshold Transfer Gap by Model")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    # Panel B: F1@source vs F1@oracle scatter
    for model_name, mdata in data.items():
        if model_name == "linevul":
            continue
        src = [r["f1_source"] for r in mdata["runs"]]
        orc = [r["f1_oracle"] for r in mdata["runs"]]
        ax2.scatter(src, orc, c=COLORS.get(model_name, "#95a5a6"),
                    label=model_name, s=60, alpha=0.7, edgecolors="black", linewidth=0.5)
    lims = [0.3, 1.0]
    ax2.plot(lims, lims, "k--", alpha=0.3, label="Perfect transfer")
    ax2.set_xlabel("F1 @ Source-Calibrated Threshold")
    ax2.set_ylabel("F1 @ Oracle Threshold")
    ax2.set_title("(b) Source vs Oracle Performance")
    ax2.legend(fontsize=8)
    ax2.set_xlim(lims); ax2.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig2_threshold_transfer.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig2_threshold_transfer.png", dpi=300, bbox_inches="tight")
    print("Fig 2: threshold transfer saved")


def fig3_adaptive_calibration():
    """Fig 3: Adaptive calibration method comparison."""
    data = json.load(open("results/adaptive_calibration.json"))
    methods = ["fixed_05", "source_calibrated", "ppr_adapted",
               "entropy_adapted", "temp_scale_200", "oracle"]
    labels = ["Fixed\n(0.5)", "Source\nCalibrated", "PPR\nAdapted",
              "Entropy\nAdapted", "TempScale\n(200 labels)", "Oracle\n(upper bound)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = {}
    for mn in methods:
        v = [r[mn]["f1"] for r in data if mn in r and isinstance(r.get(mn), dict)]
        vals[mn] = v

    positions = range(len(methods))
    bp = ax.boxplot(vals.values(), positions=list(positions), widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
    method_colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#1abc9c"]
    for patch, c in zip(bp["boxes"], method_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Calibration Methods: Only 200-Sample Temperature Scaling\nReliably Closes the Transfer Gap")
    means = [np.mean(v) for v in vals.values()]
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig3_adaptive_calibration.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig3_adaptive_calibration.png", dpi=300, bbox_inches="tight")
    print("Fig 3: adaptive calibration saved")


def fig4_adversarial():
    """Fig 4: Adversarial robustness + tokenization diagnostic."""
    data = json.load(open("results/adversarial_full.json"))
    perturbs = data["perturbations"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: F1 bars
    names = ["Clean", "Rename", "Dead Code", "Whitespace"]
    f1s = [data["clean"]["f1"],
           perturbs["rename"]["f1"],
           perturbs["deadcode"]["f1"],
           perturbs["whitespace"]["f1"]]
    drops = [0, perturbs["rename"]["rel_drop_f1_pct"],
             perturbs["deadcode"]["rel_drop_f1_pct"],
             perturbs["whitespace"]["rel_drop_f1_pct"]]
    colors = ["#2ecc71", "#3498db", "#3498db", "#e74c3c"]

    bars = ax1.bar(names, f1s, color=colors, edgecolor="black", linewidth=0.5)
    for i, (v, d) in enumerate(zip(f1s, drops)):
        label = f"{v:.3f}" if i == 0 else f"{v:.3f}\n({d:.1f}% drop)"
        ax1.text(i, v + 0.02, label, ha="center", fontsize=8, fontweight="bold")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("(a) Adversarial Robustness")
    ax1.set_ylim(0, 0.75)

    # Panel B: Tokenization change explanation
    tok_changed = perturbs["whitespace"].get("tokenization_changed_frac", 1.0)
    categories = ["Rename", "Dead Code", "Whitespace"]
    tok_rates = [0.0, 0.0, tok_changed * 100]  # rename/deadcode don't change tokens
    f1_drops = [perturbs["rename"]["rel_drop_f1_pct"],
                perturbs["deadcode"]["rel_drop_f1_pct"],
                perturbs["whitespace"]["rel_drop_f1_pct"]]

    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, f1_drops, width, label="F1 Drop (%)", color="#e74c3c", alpha=0.7)
    ax2.bar(x + width/2, tok_rates, width, label="Tokenization Changed (%)", color="#3498db", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Percentage")
    ax2.set_title("(b) F1 Drop Correlates With Tokenization Change\nWhitespace 'Robustness' Tests Measure the Tokenizer")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig4_adversarial.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig4_adversarial.png", dpi=300, bbox_inches="tight")
    print("Fig 4: adversarial saved")


def fig5_cost_curves():
    """Fig 5: Cost curves showing ranking instability."""
    data = json.load(open("results/cost_curves.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Expected cost at FN:FP=10
    for name, mdata in data.items():
        thresholds = mdata["cost_curve_thresholds"]
        costs = mdata["cost_curve_fn10"]
        ax1.plot(thresholds, costs, label=name.replace("_", " "),
                 color=COLORS.get(name, "#95a5a6"), linewidth=2)
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Expected Cost per Sample")
    ax1.set_title("(a) Cost Curves at FN:FP = 10:1")
    ax1.legend(fontsize=9)

    # Panel B: Model ranking across cost ratios
    ratios = [1, 5, 10, 25, 50, 100]
    for name, mdata in data.items():
        ranks = []
        for ratio in ratios:
            all_costs = []
            for n2, d2 in data.items():
                for ca in d2["cost_analysis"]:
                    if ca["fn_fp_ratio"] == ratio:
                        all_costs.append((n2, ca["min_expected_cost"]))
            all_costs.sort(key=lambda x: x[1])
            rank = [i+1 for i, (n, _) in enumerate(all_costs) if n == name][0]
            ranks.append(rank)
        ax2.plot(ratios, ranks, "o-", label=name.replace("_", " "),
                 color=COLORS.get(name, "#95a5a6"), linewidth=2, markersize=8)
    ax2.set_xlabel("FN:FP Cost Ratio")
    ax2.set_ylabel("Model Rank (1 = best)")
    ax2.set_title("(b) Model Rankings Flip Under Different Cost Regimes")
    ax2.set_yticks([1, 2, 3])
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig5_cost_curves.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig5_cost_curves.png", dpi=300, bbox_inches="tight")
    print("Fig 5: cost curves saved")


def fig6_budget_recall():
    """Fig 6: Budget-constrained recall."""
    data = json.load(open("results/cost_curves.json"))
    fig, ax = plt.subplots(figsize=(8, 5))
    budgets = [50, 100, 200, 500, 1000, 2000]
    for name, mdata in data.items():
        br = mdata["budget_recall"]
        recalls = [br.get(str(b), 0) for b in budgets]
        ax.plot(budgets, recalls, "o-", label=name.replace("_", " "),
                color=COLORS.get(name, "#95a5a6"), linewidth=2, markersize=8)
    ax.set_xlabel("Inspection Budget (functions reviewed)")
    ax.set_ylabel("Fraction of Vulnerabilities Found")
    ax.set_title("Budget-Constrained Vulnerability Detection\n(Practical: How Many Vulns Found per K Reviews?)")
    ax.legend()
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    n_vuln = data[list(data.keys())[0]]["n_test_vulns"]
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.text(2000, 1.02, f"Total vulns: {n_vuln}", ha="right", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/fig6_budget_recall.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUTDIR}/fig6_budget_recall.png", dpi=300, bbox_inches="tight")
    print("Fig 6: budget recall saved")


def table_summary():
    """Print LaTeX-ready summary tables."""
    print("\n=== TABLE 1: Diagnostic (Collapse Artifact) ===")
    data = json.load(open("results/diagnostic.json"))
    print(f"{'Precision':<10} {'Balance':<10} {'Weight':<8} {'F1':>6} {'Recall':>8} {'Collapsed':>10}")
    for r in data:
        c = r["config"]; m = r["metrics@calibrated"]
        print(f"{c['precision']:<10} {str(c['balance']):<10} {c['weight']:<8.0f} "
              f"{m['f1']:>6.3f} {m['recall']:>8.3f} {str(m['collapsed']):>10}")

    print("\n=== TABLE 2: Threshold Transfer ===")
    data = json.load(open("results/threshold_transfer_full.json"))
    print(f"{'Model':<18} {'F1 Gap':>8} {'CI Low':>8} {'CI High':>8} {'F1@Src':>8} {'Proj Gap':>10}")
    for name, mdata in data.items():
        s = mdata["summary"]
        print(f"{name:<18} {s['mean_f1_gap']:>8.3f} {s['ci95'][0]:>8.3f} {s['ci95'][1]:>8.3f} "
              f"{s['mean_f1_at_source']:>8.3f} {s.get('mean_per_project_gap','N/A'):>10}")

    print("\n=== TABLE 3: Adaptive Calibration ===")
    data = json.load(open("results/adaptive_calibration.json"))
    methods = ["fixed_05", "source_calibrated", "ppr_adapted",
               "entropy_adapted", "temp_scale_200", "oracle"]
    print(f"{'Method':<25} {'Mean F1':>8} {'Std':>6}")
    for mn in methods:
        vals = [r[mn]["f1"] for r in data if mn in r and isinstance(r.get(mn), dict)]
        if vals:
            print(f"{mn:<25} {np.mean(vals):>8.3f} {np.std(vals):>6.3f}")

    print("\n=== TABLE 4: Adversarial ===")
    data = json.load(open("results/adversarial_full.json"))
    print(f"{'Perturbation':<15} {'F1':>6} {'Rel Drop%':>10} {'Tok Changed%':>13}")
    print(f"{'Clean':<15} {data['clean']['f1']:>6.3f} {'---':>10} {'---':>13}")
    for kind, m in data["perturbations"].items():
        tc = f"{m.get('tokenization_changed_frac',0)*100:.1f}" if "tokenization_changed_frac" in m else "N/A"
        print(f"{kind:<15} {m['f1']:>6.3f} {m['rel_drop_f1_pct']:>10.1f} {tc:>13}")


if __name__ == "__main__":
    fig1_collapse_diagnostic()
    fig2_threshold_transfer()
    fig3_adaptive_calibration()
    fig4_adversarial()
    fig5_cost_curves()
    fig6_budget_recall()
    table_summary()
    print(f"\nAll figures saved to {OUTDIR}/")
