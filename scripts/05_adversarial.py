#!/usr/bin/env python
"""05 - ADVERSARIAL ROBUSTNESS, precisely specified + relative-drop reporting +
the tokenization-diff diagnostic that explains the whitespace anomaly.

For each perturbation: report absolute F1 AND relative drop vs clean F1
(Reviewer 2). For whitespace: report the fraction of samples whose tokenization
actually changed — if high, the 0.513 was tokenizer drift, not model fragility.
"""
import argparse, json, os
import pandas as pd
from vulnguard.data.preprocess import random_split
from vulnguard.eval.adversarial import PERTURBATIONS, apply_perturbation, tokenization_changed
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("adversarial")

def perturb_df(df, kind, seed):
    out = df.copy()
    out["func"] = [apply_perturbation(c, kind, seed=seed) for c in df["func"]]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="vulnguard")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-variants", type=int, default=1)
    ap.add_argument("--out", default="results/adversarial.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(args.seed)
    tr, va, te = random_split(df, seed=args.seed)
    is_vg = (args.model == "vulnguard")
    built = build_model(args.model)
    cfg = TrainConfig(epochs=args.epochs, balance=True, loss="weighted_ce",
                      loss_kwargs={"w0": 1.0, "w1": 30.0 if is_vg else 1.0}, seed=args.seed)
    t = Trainer(built, cfg).fit(tr, va)
    y_va, p_va, _ = t.predict(va)
    thr = pick_threshold_max_f1(y_va, p_va)

    y0, p0, _ = t.predict(te)
    clean = compute_metrics(y0, p0, thr)
    res = {"model": args.model, "is_proxy": built.is_proxy, "clean": clean, "perturbations": {}}

    for kind in PERTURBATIONS:
        f1s = []
        tok_changed = 0; total = 0
        for v in range(args.n_variants):
            pte = perturb_df(te, kind, seed=args.seed + v)
            yk, pk, _ = t.predict(pte)
            f1s.append(compute_metrics(yk, pk, thr)["f1"])
            if kind == "whitespace":
                for a, b in zip(te["func"], pte["func"]):
                    total += 1
                    tok_changed += int(tokenization_changed(built.tokenizer, a, b))
        f1 = sum(f1s) / len(f1s)
        entry = {"f1": f1, "abs_drop": clean["f1"] - f1,
                 "rel_drop_pct": 100 * (clean["f1"] - f1) / max(clean["f1"], 1e-9)}
        if kind == "whitespace" and total:
            entry["tokenization_changed_frac"] = tok_changed / total
        res["perturbations"][kind] = entry
        log.info("%s: F1=%.3f rel_drop=%.1f%%", kind, f1, entry["rel_drop_pct"])
    json.dump(res, open(args.out, "w"), indent=2)
    log.info("Wrote %s", args.out)

if __name__ == "__main__":
    main()
