#!/usr/bin/env python3
"""Full threshold-transfer study across models and seeds.

For each (model, seed, fold):
  - Train on source projects, calibrate on source val
  - Apply source threshold to target -> F1_source
  - Find oracle threshold on target -> F1_oracle
  - Compute per-project optimal thresholds in target
  - Log everything for the paper's main table + figures
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vulnguard.data.preprocess import cross_project_folds
from vulnguard.eval.metrics import (compute_metrics, pick_threshold_max_f1,
                                     pick_threshold_at_precision, bootstrap_ci)
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("transfer")

MODELS = {
    "codebert":      dict(hf="codebert",      balance=False, w1=1.0),
    "codebert_cs":   dict(hf="codebert",      balance=False, w1=10.0),   # cost-sensitive
    "graphcodebert": dict(hf="graphcodebert",  balance=False, w1=1.0),
    "linevul":       dict(hf="linevul",        balance=False, w1=1.0),
}


def run_fold(df_train_full, df_test, model_spec, fold_i, seed, epochs):
    # Split source into train + val BY PROJECT
    src_projects = df_train_full.project.unique()
    rng = np.random.default_rng(seed + fold_i)
    val_projs = set(rng.choice(src_projects, size=max(1, len(src_projects) // 5), replace=False))
    va = df_train_full[df_train_full.project.isin(val_projs)]
    tr = df_train_full[~df_train_full.project.isin(val_projs)]

    if va.label.sum() < 3 or va.label.nunique() < 2:
        tr, va = train_test_split(df_train_full, test_size=0.2,
                                  stratify=df_train_full.label, random_state=seed)

    set_seed(seed)
    built = build_model(model_spec["hf"])
    cfg = TrainConfig(epochs=epochs, balance=model_spec["balance"],
                      loss="weighted_ce",
                      loss_kwargs={"w0": 1.0, "w1": model_spec["w1"]},
                      seed=seed)
    t = Trainer(built, cfg).fit(tr, va)

    y_va, p_va, _ = t.predict(va)
    y_te, p_te, _ = t.predict(df_test)

    thr_source = pick_threshold_max_f1(y_va, p_va)
    thr_oracle = pick_threshold_max_f1(y_te, p_te)

    m_source = compute_metrics(y_te, p_te, thr_source)
    m_oracle = compute_metrics(y_te, p_te, thr_oracle)
    m_fixed = compute_metrics(y_te, p_te, 0.5)

    # Per-project analysis
    per_project = []
    for proj in df_test.project.unique():
        mask = df_test.project.values == proj
        yl, pl = y_te[mask], p_te[mask]
        if yl.sum() >= 2 and len(yl) >= 10:
            ppt = pick_threshold_max_f1(yl, pl)
            pm_src = compute_metrics(yl, pl, thr_source)
            pm_oracle = compute_metrics(yl, pl, ppt)
            per_project.append({
                "project": proj, "n": int(len(yl)), "vuln_rate": float(yl.mean()),
                "thr_optimal": ppt,
                "f1_at_source_thr": pm_src["f1"],
                "f1_at_project_oracle": pm_oracle["f1"],
                "f1_gap": pm_oracle["f1"] - pm_src["f1"],
            })

    return {
        "fold": fold_i, "seed": seed,
        "n_train": len(tr), "n_test": len(df_test),
        "test_projects": int(df_test.project.nunique()),
        "test_vuln_rate": float(y_te.mean()),
        "thr_source": thr_source, "thr_oracle": thr_oracle,
        "thr_gap": abs(thr_oracle - thr_source),
        "f1_fixed_05": m_fixed["f1"],
        "f1_source": m_source["f1"], "f1_oracle": m_oracle["f1"],
        "f1_gap": m_oracle["f1"] - m_source["f1"],
        "recall_source": m_source["recall"], "recall_oracle": m_oracle["recall"],
        "precision_source": m_source["precision"],
        "per_project_thr_std": float(np.std([p["thr_optimal"] for p in per_project])) if per_project else None,
        "per_project_f1_gap_mean": float(np.mean([p["f1_gap"] for p in per_project])) if per_project else None,
        "per_project_f1_gap_max": float(np.max([p["f1_gap"] for p in per_project])) if per_project else None,
        "n_projects_evaluated": len(per_project),
        "n_projects_with_gap_gt_01": sum(1 for p in per_project if p["f1_gap"] > 0.1) if per_project else 0,
        "per_project_details": per_project[:20],  # top 20 for analysis
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="results/threshold_transfer.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_results = {}

    for model_name in args.models:
        if model_name not in MODELS:
            log.warning("Unknown model %s, skipping", model_name)
            continue
        spec = MODELS[model_name]
        model_results = []

        for seed in args.seeds:
            for fold_i, (tr, te) in enumerate(cross_project_folds(df, args.folds, seed=seed)):
                log.info("=== %s seed=%d fold=%d ===", model_name, seed, fold_i)
                result = run_fold(tr, te, spec, fold_i, seed, args.epochs)
                result["model"] = model_name
                model_results.append(result)

                log.info("%s s%d f%d: thr_gap=%.3f F1_gap=%.3f F1@src=%.3f F1@oracle=%.3f proj_gap_mean=%s",
                         model_name, seed, fold_i,
                         result["thr_gap"], result["f1_gap"],
                         result["f1_source"], result["f1_oracle"],
                         f"{result['per_project_f1_gap_mean']:.3f}" if result["per_project_f1_gap_mean"] else "N/A")

        # Aggregate
        f1_gaps = [r["f1_gap"] for r in model_results]
        f1_src = [r["f1_source"] for r in model_results]
        proj_gaps = [r["per_project_f1_gap_mean"] for r in model_results if r["per_project_f1_gap_mean"]]
        mean_gap, lo, hi = bootstrap_ci(f1_gaps) if len(f1_gaps) >= 3 else (np.mean(f1_gaps), 0, 0)

        all_results[model_name] = {
            "runs": model_results,
            "summary": {
                "mean_f1_gap": mean_gap, "ci95": [lo, hi],
                "mean_f1_at_source": float(np.mean(f1_src)),
                "mean_per_project_gap": float(np.mean(proj_gaps)) if proj_gaps else None,
                "n_runs": len(model_results),
            }
        }
        log.info("=== %s SUMMARY: mean_F1_gap=%.3f [%.3f, %.3f] mean_F1@src=%.3f ===",
                 model_name, mean_gap, lo, hi, np.mean(f1_src))

        json.dump(all_results, open(args.out, "w"), indent=2)

    # Cross-model comparison
    log.info("=" * 70)
    log.info("CROSS-MODEL THRESHOLD TRANSFER SUMMARY:")
    for mn, mr in all_results.items():
        s = mr["summary"]
        log.info("  %s: F1_gap=%.3f [%.3f,%.3f] | F1@src=%.3f | proj_gap=%s",
                 mn, s["mean_f1_gap"], s["ci95"][0], s["ci95"][1],
                 s["mean_f1_at_source"],
                 f"{s['mean_per_project_gap']:.3f}" if s["mean_per_project_gap"] else "N/A")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
