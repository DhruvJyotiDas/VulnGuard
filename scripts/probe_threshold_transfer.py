#!/usr/bin/env python3
"""PROBE: Does the F1-optimal threshold transfer across projects?

For each cross-project fold:
  1. Train CodeBERT on source projects
  2. Calibrate threshold on source validation (held-out source projects)
  3. Apply source threshold to TARGET (unseen) projects -> F1_source
  4. Find oracle-optimal threshold on target -> F1_oracle
  5. Gap = F1_oracle - F1_source

If gap is consistently large (>0.05), Direction B is real:
  the operating point doesn't transfer, and that's the actual deployment
  problem nobody's studying.

If gap is small, the threshold transfers fine and we pivot to Direction C.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from vulnguard.data.preprocess import cross_project_folds, hard_balance
from vulnguard.eval.metrics import (compute_metrics, pick_threshold_max_f1,
                                     pick_threshold_at_precision)
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("probe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_threshold.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(args.seed)

    results = []

    for fold_i, (tr_full, te) in enumerate(cross_project_folds(df, args.folds, seed=args.seed)):
        # Split source into train + val BY PROJECT (no leak into calibration)
        src_projects = tr_full.project.unique()
        rng = np.random.default_rng(args.seed + fold_i)
        val_projs = set(rng.choice(src_projects, size=max(1, len(src_projects) // 5), replace=False))
        va = tr_full[tr_full.project.isin(val_projs)]
        tr = tr_full[~tr_full.project.isin(val_projs)]

        if va.label.sum() < 3 or va.label.nunique() < 2:
            log.warning("Fold %d: val set too small or no positives, using 20%% random split", fold_i)
            from sklearn.model_selection import train_test_split
            tr, va = train_test_split(tr_full, test_size=0.2, stratify=tr_full.label,
                                      random_state=args.seed)

        log.info("Fold %d: train=%d (%d proj) val=%d (%d proj) test=%d (%d proj)",
                 fold_i, len(tr), tr.project.nunique(), len(va), va.project.nunique(),
                 len(te), te.project.nunique())

        # Train vanilla CodeBERT (no tricks — we want to see raw transfer)
        set_seed(args.seed)
        built = build_model("codebert")
        cfg = TrainConfig(epochs=args.epochs, balance=False, loss="weighted_ce",
                          loss_kwargs={"w0": 1.0, "w1": 1.0}, seed=args.seed)
        t = Trainer(built, cfg).fit(tr, va)

        # Get predictions on val (source) and test (target)
        y_va, p_va, _ = t.predict(va)
        y_te, p_te, _ = t.predict(te)

        # Source-calibrated threshold (what you'd deploy with)
        thr_source = pick_threshold_max_f1(y_va, p_va)
        # Oracle threshold on target (what you'd WANT but can't know)
        thr_oracle = pick_threshold_max_f1(y_te, p_te)

        # Metrics at each threshold
        m_source = compute_metrics(y_te, p_te, thr_source)
        m_oracle = compute_metrics(y_te, p_te, thr_oracle)
        m_fixed = compute_metrics(y_te, p_te, 0.5)

        # Also check: what threshold gives acceptable precision (e.g. 0.3)?
        thr_p30_src = pick_threshold_at_precision(y_va, p_va, 0.3)
        thr_p30_tgt = pick_threshold_at_precision(y_te, p_te, 0.3)
        m_p30_src = compute_metrics(y_te, p_te, thr_p30_src)
        m_p30_tgt = compute_metrics(y_te, p_te, thr_p30_tgt)

        # Per-project threshold variance: what's the optimal threshold
        # for individual target projects?
        per_project_thresholds = []
        for proj in te.project.unique():
            proj_mask = te.project.values == proj
            proj_labels = y_te[proj_mask]
            proj_probs = p_te[proj_mask]
            if proj_labels.sum() >= 2 and len(proj_labels) >= 10:
                ppt = pick_threshold_max_f1(proj_labels, proj_probs)
                per_project_thresholds.append(ppt)

        fold_result = {
            "fold": fold_i,
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "test_vuln_rate": float(y_te.mean()),
            "threshold_source": thr_source,
            "threshold_oracle": thr_oracle,
            "threshold_gap": abs(thr_oracle - thr_source),
            "f1_at_fixed_05": m_fixed["f1"],
            "f1_at_source_thr": m_source["f1"],
            "f1_at_oracle_thr": m_oracle["f1"],
            "f1_gap": m_oracle["f1"] - m_source["f1"],
            "recall_at_source": m_source["recall"],
            "recall_at_oracle": m_oracle["recall"],
            "recall_gap": m_oracle["recall"] - m_source["recall"],
            "precision_at_source": m_source["precision"],
            "precision_at_oracle": m_oracle["precision"],
            # Precision-constrained operating point transfer
            "f1_at_p30_source_thr": m_p30_src["f1"],
            "f1_at_p30_target_thr": m_p30_tgt["f1"],
            "p30_gap": m_p30_tgt["f1"] - m_p30_src["f1"],
            # Per-project threshold variance
            "per_project_threshold_std": float(np.std(per_project_thresholds)) if per_project_thresholds else None,
            "per_project_threshold_range": (float(min(per_project_thresholds)),
                                            float(max(per_project_thresholds))) if per_project_thresholds else None,
            "n_projects_with_enough_data": len(per_project_thresholds),
        }

        results.append(fold_result)
        log.info(
            "Fold %d: thr_src=%.3f thr_oracle=%.3f gap=%.3f | "
            "F1@src=%.3f F1@oracle=%.3f F1_gap=%.3f | "
            "R@src=%.3f R@oracle=%.3f | proj_thr_std=%s",
            fold_i, thr_source, thr_oracle, fold_result["threshold_gap"],
            m_source["f1"], m_oracle["f1"], fold_result["f1_gap"],
            m_source["recall"], m_oracle["recall"],
            f"{fold_result['per_project_threshold_std']:.3f}" if fold_result["per_project_threshold_std"] else "N/A"
        )

        json.dump(results, open(args.out, "w"), indent=2)

    # Summary
    gaps = [r["f1_gap"] for r in results]
    thr_gaps = [r["threshold_gap"] for r in results]
    proj_stds = [r["per_project_threshold_std"] for r in results if r["per_project_threshold_std"]]

    log.info("=" * 70)
    log.info("VERDICT:")
    log.info("  Mean F1 gap (oracle - source): %.3f (std %.3f)", np.mean(gaps), np.std(gaps))
    log.info("  Mean threshold gap:            %.3f (std %.3f)", np.mean(thr_gaps), np.std(thr_gaps))
    if proj_stds:
        log.info("  Mean per-project thr std:      %.3f", np.mean(proj_stds))
    log.info("")
    mean_gap = np.mean(gaps)
    if mean_gap > 0.05:
        log.info("  >>> DIRECTION B IS VIABLE: threshold does NOT transfer well.")
        log.info("  >>> The deployment problem is real. Proceed with full study.")
    elif mean_gap > 0.02:
        log.info("  >>> BORDERLINE: modest transfer gap. B is possible but thin.")
        log.info("  >>> Consider combining with C (cost curves) as primary.")
    else:
        log.info("  >>> DIRECTION B IS DEAD: threshold transfers fine.")
        log.info("  >>> Pivot to Direction C (cost-curve evaluation) as primary.")

    json.dump(results, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
