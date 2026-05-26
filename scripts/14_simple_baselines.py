#!/usr/bin/env python3
"""Simple baselines (RF, LogReg) to address reviewer ask for non-DL comparison.
Uses TF-IDF on code tokens — no pretrained embeddings.
Shows that transformers genuinely outperform simple models, not just overfit."""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from vulnguard.data.preprocess import cross_project_folds, random_split
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("simple")

MODELS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=50,
                                                      n_jobs=-1, random_state=0),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=200,
                                                             max_depth=5, random_state=0),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, C=1.0,
                                                       random_state=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["random", "cross_project"], default="random")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-features", type=int, default=10000)
    ap.add_argument("--out", default="results/simple_baselines.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(args.seed)

    results = {}

    if args.mode == "random":
        tr, va, te = random_split(df, seed=args.seed)
        log.info("Random split: train=%d val=%d test=%d", len(tr), len(va), len(te))

        tfidf = TfidfVectorizer(max_features=args.max_features, token_pattern=r"(?u)\b\w+\b",
                                sublinear_tf=True)
        X_tr = tfidf.fit_transform(tr["func"])
        X_va = tfidf.transform(va["func"])
        X_te = tfidf.transform(te["func"])

        for name, model_fn in MODELS.items():
            log.info("Training %s...", name)
            clf = model_fn()
            clf.fit(X_tr, tr["label"].values)
            # Use predict_proba for threshold calibration
            if hasattr(clf, "predict_proba"):
                p_va = clf.predict_proba(X_va)[:, 1]
                p_te = clf.predict_proba(X_te)[:, 1]
                thr = pick_threshold_max_f1(va["label"].values, p_va)
                m = compute_metrics(te["label"].values, p_te, thr)
            else:
                preds = clf.predict(X_te)
                m = {"f1": f1_score(te["label"], preds),
                     "recall": recall_score(te["label"], preds),
                     "precision": precision_score(te["label"], preds)}
            results[name] = {"random_split": m}
            log.info("%s: F1=%.3f R=%.3f P=%.3f", name, m["f1"], m["recall"], m["precision"])

    elif args.mode == "cross_project":
        for name, model_fn in MODELS.items():
            fold_f1s = []
            for fold_i, (tr, te) in enumerate(cross_project_folds(df, args.folds, seed=args.seed)):
                log.info("%s fold %d: train=%d test=%d", name, fold_i, len(tr), len(te))
                tfidf = TfidfVectorizer(max_features=args.max_features,
                                        token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
                X_tr = tfidf.fit_transform(tr["func"])
                X_te = tfidf.transform(te["func"])
                clf = model_fn()
                clf.fit(X_tr, tr["label"].values)
                if hasattr(clf, "predict_proba"):
                    p_te = clf.predict_proba(X_te)[:, 1]
                    # Use source-calibrated threshold
                    p_tr_val = clf.predict_proba(X_tr)[:, 1]
                    thr = pick_threshold_max_f1(tr["label"].values, p_tr_val)
                    m = compute_metrics(te["label"].values, p_te, thr)
                else:
                    preds = clf.predict(X_te)
                    m = {"f1": f1_score(te["label"], preds),
                         "recall": recall_score(te["label"], preds),
                         "precision": precision_score(te["label"], preds)}
                fold_f1s.append(m["f1"])
                log.info("  fold %d F1=%.3f", fold_i, m["f1"])
            results[name] = {
                "cross_project_f1_mean": float(np.mean(fold_f1s)),
                "cross_project_f1_std": float(np.std(fold_f1s)),
                "per_fold_f1": fold_f1s,
            }
            log.info("%s cross-project: F1=%.3f (std=%.3f)", name,
                     np.mean(fold_f1s), np.std(fold_f1s))

    json.dump(results, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
