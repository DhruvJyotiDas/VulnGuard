"""Metrics + threshold calibration + the diagnostic that actually answers
'why was recall 0.004?'.

Key idea the reviewers force on us: a single F1 at threshold 0.5 hides whether
a model collapsed. So every eval here reports (a) metrics at a CALIBRATED
threshold chosen on validation, (b) the predicted-positive rate, and (c)
logit/probability distribution stats. If pred_pos_rate ~ 0 (CodeBERT) or ~ 1
(GraphCodeBERT), the model collapsed and the comparison is invalid."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def compute_metrics(labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "pred_pos_rate": float(preds.mean()),
        "base_pos_rate": float(labels.mean()),
    }
    try:
        out["pr_auc"] = float(average_precision_score(labels, probs))
        out["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    # explicit collapse flag for the rejoinder
    out["collapsed"] = bool(out["pred_pos_rate"] < 1e-3 or out["pred_pos_rate"] > 1 - 1e-3)
    return out


def pick_threshold_max_f1(labels, probs) -> float:
    """Choose the threshold that maximizes F1 on the given (validation) set.
    This is the honest way to compare models — fixing 0.5 penalizes
    poorly-calibrated-but-informative models and manufactures the 0.004."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    p, r, th = precision_recall_curve(labels, probs)
    f1 = 2 * p * r / (p + r + 1e-12)
    if len(th) == 0:
        return 0.5
    return float(th[max(0, np.nanargmax(f1[:-1]))])


def pick_threshold_at_precision(labels, probs, min_precision: float) -> float:
    """Lowest threshold achieving at least `min_precision` — the operating point
    a security team would actually pick. Use this to answer Reviewer 2's
    'precision 0.186 is unusable' critique honestly: show recall AT a tolerable
    precision instead of the precision at max recall."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    p, r, th = precision_recall_curve(labels, probs)
    ok = np.where(p[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return 1.0  # cannot reach that precision
    return float(th[ok[0]])


def logit_diagnostics(logits) -> Dict[str, float]:
    """Stats that reveal collapse / dead training. logits: [N, 2] array."""
    logits = np.asarray(logits, dtype=np.float64)
    diff = logits[:, 1] - logits[:, 0]
    return {
        "logit_pos_mean": float(logits[:, 1].mean()),
        "logit_neg_mean": float(logits[:, 0].mean()),
        "margin_mean": float(diff.mean()),
        "margin_std": float(diff.std()),
        # near-zero std => model outputs the same thing for everything => dead
        "is_constant": bool(diff.std() < 1e-3),
        "frac_predicts_pos@0.5": float((diff > 0).mean()),
    }


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI over per-fold scores (cross-project aggregation)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)
