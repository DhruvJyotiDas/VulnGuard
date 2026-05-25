#!/usr/bin/env python
"""00 - prepare data. Loads a dataset, preprocesses, caches a parquet so every
later script reads the same frame. Run once per dataset.

Examples:
  python scripts/00_prepare_data.py --dataset synthetic --out data/cache/synth.parquet
  python scripts/00_prepare_data.py --dataset bigvul --path /data/bigvul.csv --out data/cache/bigvul.parquet
"""
import argparse
import os

from vulnguard.data.loaders import load_dataset
from vulnguard.data.preprocess import preprocess
from vulnguard.utils.common import get_logger

log = get_logger("prepare")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bigvul", "devign", "synthetic"])
    ap.add_argument("--path", default=None, help="raw file path (not needed for synthetic)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-lines", type=int, default=5)
    ap.add_argument("--max-lines", type=int, default=500)
    ap.add_argument("--n", type=int, default=400, help="synthetic size")
    args = ap.parse_args()

    df = load_dataset(args.dataset, path=args.path, n=args.n)
    df = preprocess(df, args.min_lines, args.max_lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)
    log.info("Wrote %d rows -> %s", len(df), args.out)


if __name__ == "__main__":
    main()
