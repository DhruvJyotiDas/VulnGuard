# Limitations to state honestly in Threats to Validity

- **Dedup is exact-match only.** Near-duplicate leakage (Chakraborty) is NOT
  removed; that needs token-level MinHash/Jaccard. State it, or implement it.
- **Semantic preservation of adversarial transforms is heuristic**, not verified
  by compilation+test execution. Renaming preserves call semantics by construction
  (library calls untouched); dead code is provably no-op; whitespace is
  token-preserving by design but verified per-sample via tokenization diff.
- **Proxy models.** LineVul=function-level on its backbone (not line-level);
  VulDeBERT=DeBERTa-v3 fine-tune (no public checkpoint matches the paper).
- **Devign** has no per-function project labels → not valid for cross-project CV.
- **BigVul label noise** from automated CVE matching (already acknowledged).
- **Threshold calibration on validation** is the honest protocol but means
  reported numbers are at the calibrated operating point, not a fixed 0.5.
