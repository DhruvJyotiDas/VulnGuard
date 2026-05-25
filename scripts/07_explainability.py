#!/usr/bin/env python
"""07 - explainability beyond attention: integrated gradients + occlusion +
counterfactual sink removal. Reports aggregate counterfactual FLIP RATE, the
strongest single evidence the model keys on the vulnerability.
"""
import argparse, json, os
import pandas as pd
from vulnguard.data.preprocess import random_split
from vulnguard.eval.explain import counterfactual_sink_removal, occlusion_attributions
from vulnguard.models.factory import build_model
from vulnguard.train.trainer import Trainer, TrainConfig
from vulnguard.utils.common import get_logger, set_seed

log = get_logger("explain")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=50)
    ap.add_argument("--out", default="results/explain.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    set_seed(args.seed)
    tr, va, te = random_split(df, seed=args.seed)
    built = build_model("vulnguard")
    cfg = TrainConfig(epochs=args.epochs, balance=True, loss="weighted_ce",
                      loss_kwargs={"w0": 1.0, "w1": 30.0}, seed=args.seed)
    t = Trainer(built, cfg).fit(tr, va)

    vuln = te[te.label == 1].head(args.n_samples)
    flips, examples = 0, []
    for code in vuln["func"]:
        cf = counterfactual_sink_removal(built.model, built.tokenizer, code)
        if any(c["flipped"] for c in cf):
            flips += 1
        occ = occlusion_attributions(built.model, built.tokenizer, code)[:5]
        examples.append({"counterfactual": cf, "top_occlusion_tokens": occ})
    out = {"n": len(vuln), "counterfactual_flip_rate": flips / max(len(vuln), 1),
           "examples": examples[:10]}
    json.dump(out, open(args.out, "w"), indent=2)
    log.info("Counterfactual flip rate: %.2f over %d vuln samples",
             out["counterfactual_flip_rate"], out["n"])
    log.info("Wrote %s", args.out)

if __name__ == "__main__":
    main()
