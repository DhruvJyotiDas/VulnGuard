#!/usr/bin/env python3
"""Can we PREDICT when threshold transfer will fail?

Hypothesis: the transfer gap correlates with distributional divergence
between source and target model predictions. If true, practitioners can
compute a "deployment risk score" from unlabeled target code and decide
whether to recalibrate.

Metrics tested (all computable WITHOUT target labels):
1. KL divergence of predicted probability distributions
2. Mean prediction shift (mean P(vuln) on target vs source)
3. Prediction entropy difference
4. Kolmogorov-Smirnov statistic between source/target prob distributions

Correlate each with the actual F1 gap across all folds from script 10.
If Spearman r > 0.5, we have a deployable risk indicator.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from vulnguard.data.preprocess import cross_project_folds
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("transfer_pred")


def prediction_divergence_metrics(probs_source, probs_target):
    """Compute distributional divergence metrics between source and target
    predicted probabilities. All are label-free on the target side."""

    # 1. Mean prediction shift
    mean_shift = abs(float(np.mean(probs_target) - np.mean(probs_source)))

    # 2. KL divergence (discretized)
    bins = np.linspace(0, 1, 51)
    p_src = np.histogram(probs_source, bins=bins, density=True)[0] + 1e-10
    p_tgt = np.histogram(probs_target, bins=bins, density=True)[0] + 1e-10
    p_src = p_src / p_src.sum()
    p_tgt = p_tgt / p_tgt.sum()
    kl_div = float(np.sum(p_tgt * np.log(p_tgt / p_src)))

    # 3. JS divergence (symmetric, bounded)
    m = 0.5 * (p_src + p_tgt)
    js_div = float(0.5 * np.sum(p_src * np.log(p_src / m)) +
                   0.5 * np.sum(p_tgt * np.log(p_tgt / m)))

    # 4. KS statistic
    ks_stat = float(stats.ks_2samp(probs_source, probs_target).statistic)

    # 5. Prediction entropy difference
    def entropy(p):
        p_clip = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))
    entropy_diff = abs(entropy(probs_target) - entropy(probs_source))

    # 6. Predicted positive rate difference (at fixed 0.5)
    ppr_diff = abs(float((probs_target >= 0.5).mean() - (probs_source >= 0.5).mean()))

    return {
        "mean_shift": mean_shift,
        "kl_divergence": kl_div,
        "js_divergence": js_div,
        "ks_statistic": ks_stat,
        "entropy_diff": entropy_diff,
        "ppr_diff": ppr_diff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="results/transfer_prediction.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []

    for seed in args.seeds:
        for fold_i, (tr_full, te) in enumerate(cross_project_folds(df, args.folds, seed=seed)):
            src_projects = tr_full.project.unique()
            rng = np.random.default_rng(seed + fold_i)
            val_projs = set(rng.choice(src_projects, size=max(1, len(src_projects) // 5), replace=False))
            va = tr_full[tr_full.project.isin(val_projs)]
            tr = tr_full[~tr_full.project.isin(val_projs)]
            if va.label.sum() < 3 or va.label.nunique() < 2:
                tr, va = train_test_split(tr_full, test_size=0.2,
                                          stratify=tr_full.label, random_state=seed)

            log.info("Seed %d Fold %d: train=%d val=%d test=%d (%d proj)",
                     seed, fold_i, len(tr), len(va), len(te), te.project.nunique())

            set_seed(seed)
            built = build_model("codebert")
            cfg = TrainConfig(epochs=args.epochs, balance=False, loss="weighted_ce",
                              loss_kwargs={"w0": 1.0, "w1": 1.0}, seed=seed)
            t = Trainer(built, cfg).fit(tr, va)

            y_va, p_va, _ = t.predict(va)
            y_te, p_te, _ = t.predict(te)

            thr_source = pick_threshold_max_f1(y_va, p_va)
            thr_oracle = pick_threshold_max_f1(y_te, p_te)
            m_source = compute_metrics(y_te, p_te, thr_source)
            m_oracle = compute_metrics(y_te, p_te, thr_oracle)
            f1_gap = m_oracle["f1"] - m_source["f1"]

            # Compute divergence metrics
            div = prediction_divergence_metrics(p_va, p_te)

            row = {
                "seed": seed, "fold": fold_i,
                "test_projects": int(te.project.nunique()),
                "test_vuln_rate": float(y_te.mean()),
                "f1_gap": f1_gap,
                "f1_source": m_source["f1"],
                "f1_oracle": m_oracle["f1"],
                **div,
            }
            rows.append(row)
            log.info("  F1_gap=%.3f | mean_shift=%.3f KS=%.3f JS=%.4f ppr_diff=%.3f",
                     f1_gap, div["mean_shift"], div["ks_statistic"],
                     div["js_divergence"], div["ppr_diff"])

    # Correlation analysis
    log.info("=" * 70)
    log.info("CORRELATION WITH F1 GAP (Spearman):")
    f1_gaps = [r["f1_gap"] for r in rows]
    best_metric, best_r = None, 0
    for metric in ["mean_shift", "kl_divergence", "js_divergence",
                   "ks_statistic", "entropy_diff", "ppr_diff"]:
        values = [r[metric] for r in rows]
        if len(set(values)) > 2:
            rho, pval = stats.spearmanr(values, f1_gaps)
            log.info("  %-20s: rho=%.3f  p=%.4f  %s",
                     metric, rho, pval,
                     "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "")
            if abs(rho) > abs(best_r):
                best_r = rho
                best_metric = metric
        else:
            log.info("  %-20s: insufficient variance", metric)

    log.info("")
    if best_metric and abs(best_r) > 0.4:
        log.info(">>> BEST PREDICTOR: %s (rho=%.3f)", best_metric, best_r)
        log.info(">>> DIRECTION B+ IS VIABLE: distributional divergence predicts transfer failure")
        log.info(">>> This is the deployable risk indicator.")
    elif best_metric:
        log.info(">>> Best predictor: %s (rho=%.3f) — weak correlation", best_metric, best_r)
        log.info(">>> Prediction of transfer failure is unreliable with these metrics.")
    else:
        log.info(">>> No usable predictor found.")

    json.dump(rows, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
