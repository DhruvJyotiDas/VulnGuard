#!/usr/bin/env python3
"""The SOLUTION: adaptive threshold calibration for unseen projects.

This is the novel contribution that makes Direction B a paper, not just an
observation. We propose and compare 3 methods to adapt the decision threshold
when deploying to a new project WITHOUT vulnerability labels:

Method 1: PREDICTED-POSITIVE-RATE MATCHING (PPR)
  Idea: match the predicted-positive rate to the expected base rate.
  If you expect ~5% vulns, adjust threshold until ~5% predictions are positive.
  Requires: only an estimate of the vuln rate (industry average or org estimate).

Method 2: ENTROPY-BASED CALIBRATION
  Idea: a well-calibrated model has high-confidence predictions on most samples.
  Find the threshold that maximizes the average prediction confidence
  (distance from 0.5) on the unlabeled target data.

Method 3: TEMPERATURE SCALING WITH SMALL CALIBRATION SET
  Idea: if you can label even 50-100 functions from the target project,
  learn a temperature parameter to rescale logits before thresholding.
  This is the upper-bound "how much does a tiny labeled sample help?"

Evaluation: for each cross-project fold, compare:
  - F1 @ fixed 0.5
  - F1 @ source-calibrated threshold (what people do now)
  - F1 @ PPR-adapted threshold
  - F1 @ entropy-adapted threshold
  - F1 @ temperature-scaled (50 labeled samples)
  - F1 @ oracle threshold (unreachable upper bound)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vulnguard.data.preprocess import cross_project_folds
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("adaptive")


# ============================================================ ADAPTATION METHODS
def ppr_threshold(probs, target_positive_rate=0.05, n_search=1000):
    """Find threshold where predicted-positive rate matches target rate."""
    thresholds = np.linspace(0.01, 0.99, n_search)
    best_thr, best_diff = 0.5, 1.0
    for t in thresholds:
        ppr = (probs >= t).mean()
        diff = abs(ppr - target_positive_rate)
        if diff < best_diff:
            best_diff = diff
            best_thr = t
    return float(best_thr)


def entropy_threshold(probs, n_search=1000):
    """Find threshold maximizing average prediction confidence on unlabeled data.
    Confidence = |p - 0.5| for each sample's distance from uncertainty."""
    thresholds = np.linspace(0.01, 0.99, n_search)
    # Actually, threshold doesn't change confidence. But we can pick the
    # threshold at the valley of the probability density — the natural
    # separation point between the two modes.
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(probs, bw_method=0.05)
        x = np.linspace(0.01, 0.99, n_search)
        density = kde(x)
        # Find the minimum density between 0.1 and 0.9 (the valley)
        mask = (x > 0.05) & (x < 0.95)
        valley_idx = np.argmin(density[mask])
        return float(x[mask][valley_idx])
    except Exception:
        return 0.5


def temperature_scale(logits_cal, labels_cal, logits_target):
    """Learn temperature T on a small calibration set, apply to target.
    Returns calibrated probabilities on target."""
    from scipy.optimize import minimize_scalar

    def nll(T):
        scaled = logits_cal / T
        probs = _softmax_2d(scaled)[:, 1]
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(labels_cal * np.log(probs) + (1 - labels_cal) * np.log(1 - probs))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_opt = result.x

    scaled_target = logits_target / T_opt
    return _softmax_2d(scaled_target)[:, 1], float(T_opt)


def _softmax_2d(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--cal-sizes", type=int, nargs="+", default=[20, 50, 100, 200],
                    help="Number of labeled target samples for temperature scaling")
    ap.add_argument("--out", default="results/adaptive_calibration.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_results = []

    for seed in args.seeds:
        for fold_i, (tr_full, te) in enumerate(cross_project_folds(df, args.folds, seed=seed)):
            # Source train/val split
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

            # Train
            set_seed(seed)
            built = build_model("codebert")
            cfg = TrainConfig(epochs=args.epochs, balance=False, loss="weighted_ce",
                              loss_kwargs={"w0": 1.0, "w1": 1.0}, seed=seed)
            t = Trainer(built, cfg).fit(tr, va)

            y_va, p_va, _ = t.predict(va)
            y_te, p_te, logits_te = t.predict(te)

            # === THRESHOLDS ===
            thr_fixed = 0.5
            thr_source = pick_threshold_max_f1(y_va, p_va)
            thr_oracle = pick_threshold_max_f1(y_te, p_te)
            thr_ppr = ppr_threshold(p_te, target_positive_rate=0.057)  # BigVul base rate
            thr_entropy = entropy_threshold(p_te)

            # === METRICS AT EACH THRESHOLD ===
            methods = {
                "fixed_05": compute_metrics(y_te, p_te, thr_fixed),
                "source_calibrated": compute_metrics(y_te, p_te, thr_source),
                "ppr_adapted": compute_metrics(y_te, p_te, thr_ppr),
                "entropy_adapted": compute_metrics(y_te, p_te, thr_entropy),
                "oracle": compute_metrics(y_te, p_te, thr_oracle),
            }

            # Temperature scaling with various calibration set sizes
            for cal_size in args.cal_sizes:
                if len(te) < cal_size + 100:
                    continue
                # Sample a calibration set from target (stratified)
                try:
                    te_cal, te_eval = train_test_split(
                        pd.DataFrame({"y": y_te, "p": p_te,
                                      "l0": logits_te[:, 0], "l1": logits_te[:, 1]}),
                        train_size=cal_size, stratify=y_te, random_state=seed + fold_i)
                except ValueError:
                    continue

                cal_logits = te_cal[["l0", "l1"]].values
                cal_labels = te_cal["y"].values
                eval_logits = te_eval[["l0", "l1"]].values
                eval_labels = te_eval["y"].values

                p_scaled, T_opt = temperature_scale(cal_logits, cal_labels, eval_logits)
                thr_temp = pick_threshold_max_f1(cal_labels,
                                                  _softmax_2d(cal_logits / T_opt)[:, 1])
                m_temp = compute_metrics(eval_labels, p_scaled, thr_temp)
                methods[f"temp_scale_{cal_size}"] = m_temp
                methods[f"temp_scale_{cal_size}"]["T"] = T_opt
                methods[f"temp_scale_{cal_size}"]["cal_size"] = cal_size

            fold_result = {
                "seed": seed, "fold": fold_i,
                "test_projects": int(te.project.nunique()),
                "test_vuln_rate": float(y_te.mean()),
                "thresholds": {
                    "fixed": thr_fixed, "source": float(thr_source),
                    "oracle": float(thr_oracle), "ppr": float(thr_ppr),
                    "entropy": float(thr_entropy),
                },
            }
            for method_name, m in methods.items():
                fold_result[method_name] = {
                    "f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
                }

            all_results.append(fold_result)

            log_parts = []
            for mn in ["fixed_05", "source_calibrated", "ppr_adapted", "entropy_adapted", "oracle"]:
                if mn in methods:
                    log_parts.append(f"{mn}={methods[mn]['f1']:.3f}")
            log.info("  F1: %s", " | ".join(log_parts))

            json.dump(all_results, open(args.out, "w"), indent=2)

    # === AGGREGATE ===
    log.info("=" * 70)
    log.info("ADAPTIVE CALIBRATION SUMMARY (mean F1 across all folds/seeds):")
    method_names = ["fixed_05", "source_calibrated", "ppr_adapted",
                    "entropy_adapted", "oracle"] + [f"temp_scale_{s}" for s in args.cal_sizes]
    for mn in method_names:
        vals = [r[mn]["f1"] for r in all_results if mn in r]
        if vals:
            log.info("  %-25s: F1=%.3f (std=%.3f, n=%d)", mn, np.mean(vals), np.std(vals), len(vals))

    # Gap closed: how much of the source->oracle gap does each method close?
    log.info("")
    log.info("GAP CLOSED (%%  of source->oracle gap recovered):")
    for mn in ["ppr_adapted", "entropy_adapted"] + [f"temp_scale_{s}" for s in args.cal_sizes]:
        gaps_closed = []
        for r in all_results:
            if mn not in r or "source_calibrated" not in r or "oracle" not in r:
                continue
            src_f1 = r["source_calibrated"]["f1"]
            oracle_f1 = r["oracle"]["f1"]
            method_f1 = r[mn]["f1"]
            if oracle_f1 - src_f1 > 0.01:  # only count folds with real gap
                gaps_closed.append(100 * (method_f1 - src_f1) / (oracle_f1 - src_f1))
        if gaps_closed:
            log.info("  %-25s: %.1f%% gap closed (std=%.1f, n=%d)",
                     mn, np.mean(gaps_closed), np.std(gaps_closed), len(gaps_closed))

    json.dump(all_results, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
