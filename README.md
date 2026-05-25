# VulnGuard — Revision Experiment Suite

Code to address the **Tier 1 / Tier 2** reviewer comments on EISEJ-2026-0086
(*VulnGuard: Cost-Sensitive Transformer-Based Vulnerability Detection*). Built
to run on an **AMD MI300X (ROCm)**.

## Read this first — what this is and isn't

- **Tested:** all non-GPU plumbing (loaders, balancing, cross-project splits
  with anti-leak assertions, metrics, threshold calibration, logit diagnostics,
  adversarial transforms). `pytest -q` passes (9/9).
- **NOT tested by the author of this code:** anything that touches
  `torch`/`transformers` (training, prediction, IG). It was written but never
  executed on a GPU. Expect to debug. Run the **smoke mode** first.
- **Data not included.** Download BigVul / Devign yourself (links below).
- **Faithful proxies, not exact reproductions:** `linevul`, `vulberta`,
  `vuldebert` (see `src/vulnguard/models/factory.py`). Disclose this in the
  rejoinder; do not claim exact reproductions.
- **Adversarial renaming** is rigorous only with `tree-sitter` installed; the
  regex fallback is inferior — do not report regex-mode numbers.

## Install (MI300X / ROCm)

```bash
bash setup_rocm.sh          # installs ROCm torch + this package, verifies GPU
# or manually:
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
pip install -e ".[explain,adversarial]"
```

PyTorch's ROCm build reuses the `torch.cuda` namespace, so `torch.cuda.is_available()`
returns True on the MI300X — the code is device-agnostic. Defaults to **bf16**
(CDNA3-native); do **not** switch to fp16 with `w1=30` (overflow → the same
collapse you're trying to explain). With 192GB HBM3, raise `batch_size` to
64–128 and keep `grad_accum=1`.

## Smoke test (no GPU needed for plumbing; seconds)

```bash
pytest -q
DATA=data/cache/synth.parquet bash run_all.sh   # runs every script on fake data
```

If `run_all.sh` completes on synthetic data, your wiring is correct; only then
point it at real data.

## Real data

```bash
# BigVul: https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset
python scripts/00_prepare_data.py --dataset bigvul --path /data/bigvul.csv --out data/cache/bigvul.parquet
# Devign: https://github.com/epicosy/devign  (second-dataset check only; NOT cross-project, 2 projects)
python scripts/00_prepare_data.py --dataset devign --path /data/devign.json --out data/cache/devign.parquet
```

## Reviewer comment → script

| Reviewer ask | Script | What it produces |
|---|---|---|
| "CodeBERT gets ~90% F1; your 0.004 is broken" (R1, R2) | `01_baseline_diagnostic.py` | logit stats, collapse flags, fp16-vs-bf16, calibrated-threshold metrics |
| "neither balancing nor weighting alone…" (unsupported claim) + "add simpler baselines" (R1) | `02_ablation.py` | 2×2 grid isolating balance vs weight |
| **cross-project evaluation** + explain the "Cross-Proj Mean" figure (R1) | `03_cross_project.py` | leave-projects-out CV, multi-seed, bootstrap CI, Wilcoxon test |
| compare vs LineVul/VulBERTa/VulDeBERT (R1, R2) | `04_sota_baselines.py` | all models, same protocol, balanced + 95:5 |
| under-specified adversarial + relative drop (R1, R2) | `05_adversarial.py` | precise transforms, abs+rel drop, whitespace tokenization-diff |
| "why 30×?" (R2) | `06_weight_sweep.py` | P/R/F1 frontier over weights |
| "attention isn't explanation" (R1, R2) | `07_explainability.py` | IG + occlusion + counterfactual flip-rate |
| second dataset (R1) | rerun any script with `--data …/devign.parquet` | generalization check |

See `docs/REVIEWER_MAP.md` for the full comment-by-comment plan and
`docs/THREATS.md` for limitations to state honestly.

## The experiment that decides the paper

Run `01_baseline_diagnostic.py` on real BigVul **first**. If a fairly-trained
CodeBERT (bf16, calibrated threshold) recovers a normal F1, your original
0.004 was an artifact and the "reality gap" narrative must be reframed around
**cross-project generalization** (script 03), not class imbalance. Decide the
paper's story on that result before spending GPU-weeks on the rest.
