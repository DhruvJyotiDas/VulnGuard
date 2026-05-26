#!/usr/bin/env python3
"""Direction D: Rigorous adversarial robustness + counterfactual evidence.

Goes beyond "F1 dropped under renaming" to answer:
  1. Does the model track the ACTUAL vulnerability or surface cues?
     (counterfactual: neutralize the sink -> does prediction flip?)
  2. Relative drop per perturbation type (not just absolute F1)
  3. Tokenization-diff diagnostic for whitespace anomaly
  4. Per-CWE robustness (do memory-safety vulns survive perturbation
     better than logic vulns?)

Runs on cross-project fold to match the transfer study.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from vulnguard.data.preprocess import cross_project_folds
from vulnguard.eval.adversarial import (PERTURBATIONS, apply_perturbation,
                                         tokenization_changed)
from vulnguard.eval.explain import (counterfactual_sink_removal,
                                     occlusion_attributions)
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("adversarial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-counterfactual", type=int, default=100,
                    help="Number of vuln samples for counterfactual analysis")
    ap.add_argument("--out", default="results/adversarial_full.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(args.seed)

    # Use one cross-project fold
    folds = list(cross_project_folds(df, n_splits=5, seed=args.seed))
    tr_full, te = folds[2]  # fold with many test projects

    from sklearn.model_selection import train_test_split
    src_projects = tr_full.project.unique()
    rng = np.random.default_rng(args.seed)
    val_projs = set(rng.choice(src_projects, size=max(1, len(src_projects) // 5), replace=False))
    va = tr_full[tr_full.project.isin(val_projs)]
    tr = tr_full[~tr_full.project.isin(val_projs)]
    if va.label.sum() < 3:
        tr, va = train_test_split(tr_full, test_size=0.2, stratify=tr_full.label, random_state=args.seed)

    # Train vanilla CodeBERT
    log.info("Training CodeBERT for adversarial analysis...")
    built = build_model("codebert")
    cfg = TrainConfig(epochs=args.epochs, balance=False, loss="weighted_ce",
                      loss_kwargs={"w0": 1.0, "w1": 1.0}, seed=args.seed)
    t = Trainer(built, cfg).fit(tr, va)

    y_va, p_va, _ = t.predict(va)
    thr = pick_threshold_max_f1(y_va, p_va)

    # Clean baseline
    y_te, p_te, _ = t.predict(te)
    clean_metrics = compute_metrics(y_te, p_te, thr)
    log.info("Clean: F1=%.3f R=%.3f P=%.3f", clean_metrics["f1"],
             clean_metrics["recall"], clean_metrics["precision"])

    results = {"clean": clean_metrics, "threshold": float(thr)}

    # === ADVERSARIAL PERTURBATIONS ===
    perturbation_results = {}
    for kind in PERTURBATIONS:
        log.info("Applying %s perturbation...", kind)
        te_perturbed = te.copy()
        te_perturbed["func"] = [apply_perturbation(c, kind, seed=args.seed)
                                for c in te["func"]]

        y_p, p_p, _ = t.predict(te_perturbed)
        m_p = compute_metrics(y_p, p_p, thr)

        entry = {
            "f1": m_p["f1"],
            "recall": m_p["recall"],
            "precision": m_p["precision"],
            "abs_drop_f1": clean_metrics["f1"] - m_p["f1"],
            "rel_drop_f1_pct": 100 * (clean_metrics["f1"] - m_p["f1"]) / max(clean_metrics["f1"], 1e-9),
            "abs_drop_recall": clean_metrics["recall"] - m_p["recall"],
        }

        # Whitespace tokenization diagnostic
        if kind == "whitespace":
            changed = sum(1 for a, b in zip(te["func"], te_perturbed["func"])
                          if tokenization_changed(built.tokenizer, a, b))
            entry["tokenization_changed_frac"] = changed / len(te)
            entry["tokenization_changed_count"] = changed
            log.info("  Whitespace: %d/%d (%.1f%%) samples had tokenization change",
                     changed, len(te), 100 * changed / len(te))

        perturbation_results[kind] = entry
        log.info("  %s: F1=%.3f (drop=%.3f, %.1f%%)", kind, m_p["f1"],
                 entry["abs_drop_f1"], entry["rel_drop_f1_pct"])

    results["perturbations"] = perturbation_results

    # === PER-CWE ADVERSARIAL ROBUSTNESS ===
    log.info("Per-CWE adversarial analysis...")
    cwe_adv = defaultdict(dict)
    for cwe in te["cwe"].value_counts().head(10).index:
        if not cwe or cwe == "":
            continue
        cwe_mask = te["cwe"].values == cwe
        if cwe_mask.sum() < 10:
            continue
        te_cwe = te[cwe_mask]
        y_c, p_c, _ = t.predict(te_cwe)
        clean_cwe = compute_metrics(y_c, p_c, thr)
        cwe_adv[cwe]["clean_f1"] = clean_cwe["f1"]

        for kind in ["rename", "deadcode"]:
            te_p = te_cwe.copy()
            te_p["func"] = [apply_perturbation(c, kind, seed=args.seed) for c in te_cwe["func"]]
            y_p, p_p, _ = t.predict(te_p)
            m_p = compute_metrics(y_p, p_p, thr)
            cwe_adv[cwe][f"{kind}_f1"] = m_p["f1"]
            cwe_adv[cwe][f"{kind}_drop"] = clean_cwe["f1"] - m_p["f1"]

    results["per_cwe_adversarial"] = dict(cwe_adv)

    # === COUNTERFACTUAL ANALYSIS ===
    log.info("Counterfactual sink removal on %d vuln samples...", args.n_counterfactual)
    vuln_samples = te[te.label == 1].head(args.n_counterfactual)

    counterfactual_results = []
    flips = 0
    total_with_sinks = 0
    for _, row in vuln_samples.iterrows():
        cf = counterfactual_sink_removal(built.model, built.tokenizer, row["func"])
        if cf:  # has at least one sink
            total_with_sinks += 1
            has_flip = any(c["flipped"] for c in cf)
            if has_flip:
                flips += 1
            counterfactual_results.append({
                "cwe": row["cwe"],
                "sinks_found": [c["sink"] for c in cf],
                "flipped": has_flip,
                "max_delta": max(c["delta"] for c in cf) if cf else 0,
                "details": cf,
            })

    results["counterfactual"] = {
        "n_tested": len(vuln_samples),
        "n_with_sinks": total_with_sinks,
        "n_flipped": flips,
        "flip_rate": flips / max(total_with_sinks, 1),
        "examples": counterfactual_results[:20],
    }
    log.info("Counterfactual: %d/%d flipped (%.1f%% of samples with sinks)",
             flips, total_with_sinks, 100 * flips / max(total_with_sinks, 1))

    # === OCCLUSION on a few examples ===
    log.info("Occlusion attribution on 10 samples...")
    occ_examples = []
    for _, row in vuln_samples.head(10).iterrows():
        occ = occlusion_attributions(built.model, built.tokenizer, row["func"])
        occ_examples.append({
            "cwe": row["cwe"],
            "top_tokens": occ[:10],
        })
    results["occlusion_examples"] = occ_examples

    json.dump(results, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
