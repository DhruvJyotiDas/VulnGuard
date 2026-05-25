#!/usr/bin/env python
"""06 - class-weight sweep. Justifies the 30x choice (reviewers: 'why 30?').
Trains CodeBERT at w1 in {1,10,20,30,50,100} and reports the precision-recall
frontier on the imbalanced (95:5) test set. Pick the knee, not 30 by fiat.
"""
import argparse, json, os
import pandas as pd
from vulnguard.data.preprocess import make_imbalanced_test, random_split
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("sweep")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=[1, 10, 20, 30, 50, 100])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/weight_sweep.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for w in args.weights:
        set_seed(args.seed)
        tr, va, te = random_split(df, seed=args.seed)
        te_imb = make_imbalanced_test(te, ratio=0.05, seed=args.seed)
        built = build_model("codebert")
        cfg = TrainConfig(epochs=args.epochs, balance=True, loss="weighted_ce",
                          loss_kwargs={"w0": 1.0, "w1": w}, seed=args.seed)
        t = Trainer(built, cfg).fit(tr, va)
        y_va, p_va, _ = t.predict(va)
        thr = pick_threshold_max_f1(y_va, p_va)
        y_i, p_i, _ = t.predict(te_imb)
        m = compute_metrics(y_i, p_i, thr)
        m["w1"] = w
        rows.append(m)
        log.info("w1=%g | P=%.3f R=%.3f F1=%.3f", w, m["precision"], m["recall"], m["f1"])
        json.dump(rows, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)

if __name__ == "__main__":
    main()
