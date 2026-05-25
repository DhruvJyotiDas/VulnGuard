#!/usr/bin/env python
"""03 - CROSS-PROJECT GENERALIZATION (the experiment that actually supports
'real-world deployment' — Reviewer 1's central demand, and the one that explains
the mysterious 'Cross-Proj Mean' numbers in the original Figure 2).

Leave-projects-out CV: train on a set of projects, test on UNSEEN projects, so
nothing leaks. Repeat over folds and seeds, aggregate with bootstrap CIs, and
run a Wilcoxon signed-rank test of VulnGuard vs the vanilla CodeBERT baseline
across paired folds.

Compares two configs by default: vanilla CodeBERT (balance off, 1x) and
VulnGuard (balance on, 30x). Add LineVul/VulBERTa by extending CONFIGS.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from vulnguard.data.preprocess import cross_project_folds
from vulnguard.eval.metrics import bootstrap_ci, compute_metrics, pick_threshold_max_f1
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("crossproj")

CONFIGS = {
    "codebert_vanilla": dict(model="codebert", balance=False, w1=1.0),
    "vulnguard":        dict(model="codebert", balance=True,  w1=30.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="results/cross_project.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # per-config list of per-(seed,fold) F1 on the unseen-project test set
    f1s = defaultdict(list)
    recalls = defaultdict(list)
    raw = []

    for seed in args.seeds:
        for fold_i, (tr_full, te) in enumerate(cross_project_folds(df, args.folds, seed=seed)):
            # carve a val split off train by PROJECT too (avoid leak into calibration)
            val_projects = pd.Series(tr_full.project.unique()).sample(
                frac=0.15, random_state=seed)
            va = tr_full[tr_full.project.isin(val_projects)]
            tr = tr_full[~tr_full.project.isin(val_projects)]
            if len(va) == 0 or va.label.nunique() < 2:
                va = tr_full  # fallback; small datasets

            for cfg_name, spec in CONFIGS.items():
                set_seed(seed)
                built = build_model(spec["model"])
                cfg = TrainConfig(epochs=args.epochs, balance=spec["balance"],
                                  loss="weighted_ce",
                                  loss_kwargs={"w0": 1.0, "w1": spec["w1"]}, seed=seed)
                t = Trainer(built, cfg).fit(tr, va)
                y_va, p_va, _ = t.predict(va)
                thr = pick_threshold_max_f1(y_va, p_va)
                y_te, p_te, _ = t.predict(te)
                m = compute_metrics(y_te, p_te, thr)
                f1s[cfg_name].append(m["f1"])
                recalls[cfg_name].append(m["recall"])
                raw.append({"seed": seed, "fold": fold_i, "config": cfg_name, **m})
                log.info("seed=%d fold=%d %s F1=%.3f recall=%.3f",
                         seed, fold_i, cfg_name, m["f1"], m["recall"])

    summary = {}
    for cfg_name in CONFIGS:
        mean, lo, hi = bootstrap_ci(f1s[cfg_name])
        rmean, rlo, rhi = bootstrap_ci(recalls[cfg_name])
        summary[cfg_name] = {
            "f1_mean": mean, "f1_ci95": [lo, hi],
            "recall_mean": rmean, "recall_ci95": [rlo, rhi],
            "n_runs": len(f1s[cfg_name]),
        }

    # paired test: vulnguard vs codebert_vanilla on matched (seed,fold)
    if {"vulnguard", "codebert_vanilla"} <= set(f1s):
        a, b = np.array(f1s["vulnguard"]), np.array(f1s["codebert_vanilla"])
        if len(a) == len(b) and len(a) >= 5 and np.any(a != b):
            stat, p = wilcoxon(a, b)
            summary["wilcoxon_vulnguard_vs_vanilla"] = {"statistic": float(stat), "p_value": float(p)}

    json.dump({"summary": summary, "raw": raw}, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)
    log.info("CROSS-PROJECT F1 (the number that matters): %s",
             {k: round(v["f1_mean"], 3) for k, v in summary.items() if "f1_mean" in v})


if __name__ == "__main__":
    main()
