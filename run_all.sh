#!/usr/bin/env bash
# End-to-end driver. Swap DATA for your real BigVul/Devign parquet.
# SMOKE first (CPU, seconds) to catch plumbing bugs before burning GPU hours.
set -euo pipefail
DATA=${DATA:-data/cache/synth.parquet}

if [ ! -f "$DATA" ]; then
  echo "[run_all] preparing synthetic smoke data"
  python scripts/00_prepare_data.py --dataset synthetic --n 600 --out "$DATA"
fi

python scripts/01_baseline_diagnostic.py --data "$DATA" --epochs 2
python scripts/02_ablation.py            --data "$DATA" --epochs 2
python scripts/03_cross_project.py       --data "$DATA" --folds 4 --seeds 0 1 --epochs 2
python scripts/04_sota_baselines.py      --data "$DATA" --epochs 2 --models codebert vulnguard
python scripts/05_adversarial.py         --data "$DATA" --epochs 2 --model vulnguard
python scripts/06_weight_sweep.py        --data "$DATA" --epochs 2 --weights 1 30 100
python scripts/07_explainability.py      --data "$DATA" --epochs 2 --n-samples 10
echo "[run_all] done -> results/"
