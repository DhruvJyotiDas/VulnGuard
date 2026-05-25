#!/usr/bin/env python
"""01 - BASELINE DIAGNOSTIC.  The most important script in this repo.

Reviewer 1 & 2 both say: CodeBERT gets ~90% F1 on BigVul, your 0.004 is a broken
baseline. This script tests *why* the original collapsed, by training CodeBERT
under controlled conditions and dumping the evidence:

  - logit diagnostics (is the model constant? -> dead training)
  - metrics at threshold 0.5 vs a CALIBRATED threshold (is it just mis-thresholded?)
  - fp16 vs bf16 (does precision cause the collapse under weighted loss?)
  - balanced-train vs imbalanced-train (does the original setup escape the prior?)

Run the grid and read results/diagnostic.json. If CodeBERT recovers (F1 >> 0.008)
under bf16 + calibrated threshold, your original 0.004 was an artifact and the
paper's headline must be reframed. That is the single finding that decides the
paper.
"""
import argparse
import itertools
import json
import os

import pandas as pd

from vulnguard.data.preprocess import random_split
from vulnguard.eval.metrics import (compute_metrics, logit_diagnostics,
                                     pick_threshold_max_f1)
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("diagnostic")


def run_one(df, *, precision, balance, weight, epochs, seed):
    set_seed(seed)
    tr, va, te = random_split(df, seed=seed)
    built = build_model("codebert")
    cfg = TrainConfig(
        epochs=epochs, prefer_bf16=(precision == "bf16"),
        balance=balance, loss="weighted_ce", loss_kwargs={"w0": 1.0, "w1": weight},
        seed=seed,
    )
    tr_ = Trainer(built, cfg).fit(tr, va)
    # calibrate threshold on validation
    y_va, p_va, _ = tr_.predict(va)
    thr = pick_threshold_max_f1(y_va, p_va)
    y_te, p_te, logits_te = tr_.predict(te)
    return {
        "config": {"precision": precision, "balance": balance, "weight": weight,
                   "epochs": epochs, "seed": seed},
        "metrics@0.5": compute_metrics(y_te, p_te, 0.5),
        "metrics@calibrated": compute_metrics(y_te, p_te, thr),
        "logit_diagnostics": logit_diagnostics(logits_te),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet from 00_prepare_data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/diagnostic.json")
    ap.add_argument("--full-grid", action="store_true",
                    help="run all precision x balance x weight combos (slow)")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.full_grid:
        grid = list(itertools.product(["bf16", "fp16"], [True, False], [1.0, 30.0]))
    else:
        # the four cells that matter most: reproduce-collapse vs fixed
        grid = [("fp16", False, 30.0),   # closest to the original (suspected broken)
                ("bf16", False, 30.0),   # same but bf16
                ("bf16", True, 30.0),    # VulnGuard-style
                ("bf16", True, 1.0)]     # fair vanilla CodeBERT baseline
    results = []
    for precision, balance, weight in grid:
        log.info("=== precision=%s balance=%s weight=%s ===", precision, balance, weight)
        results.append(run_one(df, precision=precision, balance=balance,
                               weight=weight, epochs=args.epochs, seed=args.seed))
        json.dump(results, open(args.out, "w"), indent=2)
    log.info("Wrote %s. Inspect 'collapsed' flags and metrics@calibrated.", args.out)


if __name__ == "__main__":
    main()
