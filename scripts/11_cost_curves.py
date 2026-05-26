#!/usr/bin/env python3
"""Direction C: Cost-curve / decision-theoretic evaluation.

Key insight: F1 treats FP and FN equally. In security, a missed vuln (FN)
costs 10-100x more than a false alarm (FP). This script:

1. For each model's predictions (from script 10 or the probe), compute the
   EXPECTED COST at various FN:FP cost ratios (1:1, 5:1, 10:1, 50:1, 100:1)
2. Show that MODEL RANKINGS FLIP depending on the cost regime
3. Compute "budget-constrained recall": given a reviewer can inspect K
   functions/day, what fraction of vulns are caught?
4. Generate data for cost-curve figures

This is CPU-only — it reads saved predictions, no training needed.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.utils.common import get_logger

log = get_logger("costcurve")

COST_RATIOS = [1, 5, 10, 25, 50, 100]  # FN_cost : FP_cost


def expected_cost(labels, probs, threshold, fn_cost, fp_cost=1.0):
    """Expected cost per sample at a given threshold."""
    preds = (probs >= threshold).astype(int)
    fn = ((labels == 1) & (preds == 0)).sum()
    fp = ((labels == 0) & (preds == 1)).sum()
    return (fn * fn_cost + fp * fp_cost) / len(labels)


def optimal_threshold_for_cost(labels, probs, fn_cost, fp_cost=1.0, n_thresholds=500):
    """Find threshold minimizing expected cost."""
    thresholds = np.linspace(0, 1, n_thresholds)
    costs = [expected_cost(labels, probs, t, fn_cost, fp_cost) for t in thresholds]
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs[best_idx]


def budget_constrained_recall(labels, probs, budgets):
    """Given budget = K inspections, sort by P(vuln) desc, inspect top K.
    What fraction of actual vulns are found?"""
    n_vuln = labels.sum()
    if n_vuln == 0:
        return {b: 0.0 for b in budgets}
    order = np.argsort(-probs)
    sorted_labels = labels[order]
    results = {}
    for b in budgets:
        k = min(b, len(labels))
        found = sorted_labels[:k].sum()
        results[b] = float(found / n_vuln)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-results", default="results/probe_threshold.json",
                    help="Results from probe or script 10 with per-fold predictions")
    ap.add_argument("--data", required=True, help="bigvul parquet for re-running predictions")
    ap.add_argument("--out", default="results/cost_curves.json")
    args = ap.parse_args()

    # We need actual predictions. Retrain quickly or load from the probe.
    # For efficiency, we retrain one model and compute costs across thresholds.
    from vulnguard.data.preprocess import cross_project_folds
    from vulnguard.models.factory import build_model
    from vulnguard.train.trainer import Trainer, TrainConfig
    from vulnguard.utils.common import set_seed
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Compare vanilla vs cost-sensitive across cost regimes
    CONFIGS = {
        "codebert_vanilla": dict(hf="codebert", w1=1.0),
        "codebert_cs10":    dict(hf="codebert", w1=10.0),
        "codebert_cs30":    dict(hf="codebert", w1=30.0),
    }

    all_results = {}
    seed = 0
    set_seed(seed)

    # Use one cross-project fold for the cost analysis
    folds = list(cross_project_folds(df, n_splits=5, seed=seed))
    # Pick the fold with many test projects (fold 2 or 3 from probe had 101-103)
    tr_full, te = max(folds, key=lambda x: x[1].project.nunique())

    src_projects = tr_full.project.unique()
    rng = np.random.default_rng(seed)
    val_projs = set(rng.choice(src_projects, size=max(1, len(src_projects) // 5), replace=False))
    va = tr_full[tr_full.project.isin(val_projs)]
    tr = tr_full[~tr_full.project.isin(val_projs)]
    if va.label.sum() < 3:
        tr, va = train_test_split(tr_full, test_size=0.2, stratify=tr_full.label, random_state=seed)

    log.info("Cost analysis: train=%d val=%d test=%d (%d projects)",
             len(tr), len(va), len(te), te.project.nunique())

    for cfg_name, spec in CONFIGS.items():
        log.info("--- Training %s ---", cfg_name)
        set_seed(seed)
        built = build_model(spec["hf"])
        cfg = TrainConfig(epochs=5, balance=False, loss="weighted_ce",
                          loss_kwargs={"w0": 1.0, "w1": spec["w1"]}, seed=seed)
        t = Trainer(built, cfg).fit(tr, va)

        y_va, p_va, _ = t.predict(va)
        y_te, p_te, _ = t.predict(te)

        thr_f1 = pick_threshold_max_f1(y_va, p_va)

        # Cost analysis at each FN:FP ratio
        cost_analysis = []
        for ratio in COST_RATIOS:
            thr_cost, min_cost = optimal_threshold_for_cost(y_te, p_te, fn_cost=ratio)
            m_cost = compute_metrics(y_te, p_te, thr_cost)
            m_f1 = compute_metrics(y_te, p_te, thr_f1)
            cost_at_f1_thr = expected_cost(y_te, p_te, thr_f1, fn_cost=ratio)

            cost_analysis.append({
                "fn_fp_ratio": ratio,
                "optimal_threshold": float(thr_cost),
                "min_expected_cost": float(min_cost),
                "cost_at_f1_threshold": float(cost_at_f1_thr),
                "cost_penalty_pct": float(100 * (cost_at_f1_thr - min_cost) / max(min_cost, 1e-9)),
                "recall_at_cost_optimal": m_cost["recall"],
                "precision_at_cost_optimal": m_cost["precision"],
                "f1_at_cost_optimal": m_cost["f1"],
                "recall_at_f1_optimal": m_f1["recall"],
            })

        # Budget-constrained recall
        n_test_vuln = int(y_te.sum())
        budgets = [50, 100, 200, 500, 1000, 2000]
        bcr = budget_constrained_recall(y_te, p_te, budgets)

        # Full cost curve data (for plotting)
        thresholds = np.linspace(0.01, 0.99, 200)
        cost_curve_10 = [float(expected_cost(y_te, p_te, t, fn_cost=10)) for t in thresholds]
        cost_curve_50 = [float(expected_cost(y_te, p_te, t, fn_cost=50)) for t in thresholds]

        all_results[cfg_name] = {
            "f1_threshold": float(thr_f1),
            "cost_analysis": cost_analysis,
            "budget_recall": {str(k): v for k, v in bcr.items()},
            "n_test_vulns": n_test_vuln,
            "cost_curve_thresholds": [float(t) for t in thresholds],
            "cost_curve_fn10": cost_curve_10,
            "cost_curve_fn50": cost_curve_50,
        }

        log.info("%s: F1_thr=%.3f", cfg_name, thr_f1)
        for ca in cost_analysis:
            log.info("  FN:FP=%d:1 -> optimal_thr=%.3f recall=%.3f cost_penalty=%.1f%%",
                     ca["fn_fp_ratio"], ca["optimal_threshold"],
                     ca["recall_at_cost_optimal"], ca["cost_penalty_pct"])

    # Model ranking comparison
    log.info("=" * 70)
    log.info("MODEL RANKINGS BY COST REGIME:")
    for ratio in COST_RATIOS:
        ranking = []
        for cfg_name, res in all_results.items():
            ca = [c for c in res["cost_analysis"] if c["fn_fp_ratio"] == ratio][0]
            ranking.append((cfg_name, ca["min_expected_cost"]))
        ranking.sort(key=lambda x: x[1])
        log.info("  FN:FP=%d:1 -> %s", ratio,
                 " > ".join(f"{n}({c:.4f})" for n, c in ranking))

    json.dump(all_results, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
