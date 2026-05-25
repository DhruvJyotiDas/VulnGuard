# Comment-by-comment revision plan

Tiering and effort are from the strategic review. "Script" = the file in this
repo that generates the evidence.

## Tier 1 — fatal unless fixed

### Broken/strawman baselines (R1 "fairly configured", R2 ×3)
- Run `01_baseline_diagnostic.py` on real BigVul. Report logit diagnostics and
  calibrated-threshold metrics. If CodeBERT recovers, REFRAME the paper.
- Then run `02_ablation.py`: vanilla / weight-only / balance-only / vulnguard.
  This supplies the missing table behind "neither alone attains VulnGuard".
- Add the standard-BigVul baseline number (train+test full data) so reviewers
  can anchor against the ~90% F1 they expect.

### Cross-project generalization (R1, twice; explains Fig.2 "Cross-Proj Mean")
- `03_cross_project.py`: leave-projects-out CV, seeds {0,1,2}, bootstrap CI,
  Wilcoxon vs vanilla. THIS is what supports "real-world deployment".
- **Action on the original figure:** if the 0.017/0.142/0.351 numbers were not
  from a saved, reproducible run, DELETE them from the manuscript now and
  replace with script-03 output. Unbacked figure numbers are an integrity risk.

### SOTA baselines — LineVul/VulBERTa/VulDeBERT (R1, R2)
- `04_sota_baselines.py`. Proxies flagged. At minimum include LineVul.

### Second dataset — Devign (R1)
- Re-run 01/02/04 with `--data devign.parquet`. (Cross-project CV invalid on
  Devign: only 2 projects.)

## Tier 2 — substantive

### Adversarial spec + relative drop (R1, R2)
- `05_adversarial.py`. Document in the paper, verbatim: locals/params only;
  library calls preserved; dead code provably no-op; 1 variant/sample.
- Report relative drop, not just absolute F1.
- **Whitespace anomaly (0.513):** script logs `tokenization_changed_frac`. If
  high, state plainly the original number reflected tokenizer drift, and report
  the fixed result.

### Justify 30× (R2)
- `06_weight_sweep.py`. Show the PR frontier; pick the knee, justify the choice.

### Explainability beyond attention (R1, R2)
- `07_explainability.py`. Report counterfactual flip-rate + IG/occlusion
  convergence on the sink. Tone down attention claims to "consistent with",
  not "proves".

### Precision 0.186 critical discussion (R2)
- Use `pick_threshold_at_precision` to report recall at a tolerable precision,
  and compare the resulting FP rate to the static analyzers the intro
  criticizes (note: 0.186 precision = 81.4% FP rate = the same problem).

### Define experimental settings explicitly (R2)
- Every results table: state train vs test distribution and whether
  cost-sensitivity is on. (Scripts label balanced/imbalanced/calibrated.)

### Related-work rebuild + conceptual corrections (R1, R2)
- Add LineVul, VulBERTa, VulDeBERT, PrimeVul (verify the PrimeVul citation —
  the current author list looks mis-attributed), DiverseVul.
- Fix: false positives stem from Rice's-theorem over-approximation, not just
  "manual rules"; transformers capture lexical/textual patterns, not execution
  semantics; reframe deployment barrier as generalization, not imbalance.

### CWE scope rationale (R2)
- Justify memory-safety focus or report all 91.

## Tier 3 — cosmetic
PR-AUC consistency; F1 subscript; Fig.1 caption colors; dissolve standalone
"Problem statement/Research objective" labels into prose; §3.1 flow; §6 rewrite;
missing page-3 citations; fix A100-vs-T4 and duplicate-appendix inconsistencies.
