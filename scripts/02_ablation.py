#!/usr/bin/env python
"""02 - ABLATION (the contribution-isolation experiment).

Discussion claims 'neither balancing nor weighting alone attains VulnGuard's
performance' but the paper never runs the ablation. This does:

    cell           balance   weight
    vanilla          no        1x
    weight_only      no        30x
    balance_only     yes        1x
    vulnguard        yes       30x

Same backbone (CodeBERT), same trainer, same data. Any F1 delta is attributable
to exactly these two knobs. Report as a 2x2 table in the rejoinder.
"""
import argparse
import json
import os

import pandas as pd

from vulnguard.data.preprocess import (make_imbalanced_test, random_split)
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("ablation")

CELLS = {
    "vanilla":      dict(balance=False, w1=1.0),
    "weight_only":  dict(balance=False, w1=30.0),
    "balance_only": dict(balance=True,  w1=1.0),
    "vulnguard":    dict(balance=True,  w1=30.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = {}

    for name, spec in CELLS.items():
        set_seed(args.seed)
        tr, va, te = random_split(df, seed=args.seed)
        te_imb = make_imbalanced_test(te, ratio=0.05, seed=args.seed)
        built = build_model("codebert")
        cfg = TrainConfig(epochs=args.epochs, balance=spec["balance"],
                          loss="weighted_ce", loss_kwargs={"w0": 1.0, "w1": spec["w1"]},
                          seed=args.seed)
        t = Trainer(built, cfg).fit(tr, va)
        y_va, p_va, _ = t.predict(va)
        thr = pick_threshold_max_f1(y_va, p_va)
        y_b, p_b, _ = t.predict(te)
        y_i, p_i, _ = t.predict(te_imb)
        results[name] = {
            "spec": spec,
            "balanced":   compute_metrics(y_b, p_b, thr),
            "imbalanced": compute_metrics(y_i, p_i, thr),
        }
        log.info("%s | bal F1=%.3f | imb F1=%.3f recall=%.3f",
                 name, results[name]["balanced"]["f1"],
                 results[name]["imbalanced"]["f1"], results[name]["imbalanced"]["recall"])
        json.dump(results, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
