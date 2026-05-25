"""Fast unit tests — run with `pytest -q`. No GPU/network needed; they validate
data plumbing, balancing, splits, metrics, and adversarial transforms."""
import numpy as np
from vulnguard.data.loaders import load_synthetic
from vulnguard.data.preprocess import (cross_project_folds, hard_balance,
                                       make_imbalanced_test, preprocess, random_split)
from vulnguard.eval.adversarial import (apply_perturbation, rename_variables)
from vulnguard.eval.metrics import compute_metrics, pick_threshold_max_f1, logit_diagnostics


def test_synthetic_schema():
    df = load_synthetic(300, seed=0)
    assert set(["func", "label", "project", "cwe"]).issubset(df.columns)
    assert df.label.isin([0, 1]).all()
    assert df.label.sum() >= 1


def test_hard_balance_is_1to1():
    df = load_synthetic(500, vuln_ratio=0.1, seed=1)
    bal = hard_balance(df, seed=0)
    assert bal.label.sum() == (1 - bal.label).sum()


def test_cross_project_no_leak():
    df = preprocess(load_synthetic(600, n_projects=10, seed=2))
    for tr, te in cross_project_folds(df, n_splits=5, seed=0):
        assert not (set(tr.project) & set(te.project))


def test_imbalanced_test_ratio():
    df = load_synthetic(2000, vuln_ratio=0.3, seed=3)
    imb = make_imbalanced_test(df, ratio=0.05, seed=0)
    assert abs(imb.label.mean() - 0.05) < 0.02


def test_metrics_collapse_flag():
    labels = np.array([0, 1] * 50)
    probs = np.zeros(100)          # always predicts negative -> collapse
    m = compute_metrics(labels, probs, 0.5)
    assert m["collapsed"] and m["recall"] == 0.0


def test_threshold_calibration_recovers_recall():
    # informative-but-shifted probs: 0.5 threshold collapses, calibration fixes it
    rng = np.random.default_rng(0)
    labels = np.array([0] * 80 + [1] * 20)
    probs = np.where(labels == 1, rng.uniform(0.2, 0.4, 100), rng.uniform(0.0, 0.2, 100))
    thr = pick_threshold_max_f1(labels, probs)
    assert compute_metrics(labels, probs, thr)["recall"] > compute_metrics(labels, probs, 0.5)["recall"]


def test_logit_diagnostics_detects_dead_model():
    logits = np.tile([2.0, -2.0], (50, 1))  # constant output
    d = logit_diagnostics(logits)
    assert d["is_constant"]


def test_rename_preserves_library_calls():
    code = "int f(char *src){ char buf[10]; strcpy(buf, src); return 0; }"
    out = rename_variables(code, seed=0)
    assert "strcpy" in out  # library call must NOT be renamed


def test_perturbations_run():
    code = "int f(int a){ int b = a + 1; return b; }"
    for kind in ["rename", "deadcode", "whitespace"]:
        assert isinstance(apply_perturbation(code, kind, seed=0), str)
