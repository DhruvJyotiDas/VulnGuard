"""Preprocessing, hard balancing, and the two split regimes the reviewers care
about: a within-project random split (the inflated one) and a cross-project
split (the realistic one)."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from ..utils.common import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------- preprocessing
def preprocess(df: pd.DataFrame, min_lines: int = 5, max_lines: int = 500,
               drop_duplicates: bool = True) -> pd.DataFrame:
    n0 = len(df)
    if drop_duplicates:
        # exact-dedup on normalized whitespace. NOTE: this does NOT remove the
        # near-duplicate leakage Chakraborty flagged; that requires token-level
        # MinHash. Flagged as a known limitation in docs/THREATS.md.
        key = df["func"].str.replace(r"\s+", " ", regex=True).str.strip()
        df = df[~key.duplicated()].copy()
    n_lines = df["func"].str.count("\n") + 1
    df = df[(n_lines >= min_lines) & (n_lines <= max_lines)].reset_index(drop=True)
    log.info("Preprocess: %d -> %d (dedup + length filter [%d,%d] lines)",
             n0, len(df), min_lines, max_lines)
    return df


# ------------------------------------------------------------------- balancing
def hard_balance(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Undersample the majority (safe) class to a 1:1 ratio. Returns a SHUFFLED
    balanced frame. Re-call per epoch with a different seed to vary the safe
    sample (the paper's 'new balanced data each epoch')."""
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    k = min(len(pos), len(neg))
    rng = np.random.default_rng(seed)
    pos_s = pos.sample(n=k, random_state=int(rng.integers(1 << 31))) if len(pos) > k else pos
    neg_s = neg.sample(n=k, random_state=int(rng.integers(1 << 31)))
    return pd.concat([pos_s, neg_s]).sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


# ---------------------------------------------------------------------- splits
def random_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1,
                 seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split. This is the LEAKY regime (functions from one
    project land in both train and test). Report it ONLY to reproduce the
    inflated literature numbers, then contrast against cross_project_folds."""
    tr, te = train_test_split(df, test_size=test_size, stratify=df.label, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_size / (1 - test_size),
                              stratify=tr.label, random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def cross_project_folds(df: pd.DataFrame, n_splits: int = 5,
                        seed: int = 0) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Leave-projects-out CV. Guarantees NO project appears in both train and
    test of a fold -> this is the experiment Reviewer 1 demands. Yields
    (train_df, test_df). Carve your own val split off train inside the loop.

    Guard: errors out if there are fewer projects than folds (e.g. Devign)."""
    n_proj = df["project"].nunique()
    if n_proj < n_splits:
        raise ValueError(f"Only {n_proj} projects but n_splits={n_splits}. "
                         f"Cross-project CV is meaningless here.")
    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold is deterministic; shuffle project assignment via seed for
    # multi-seed runs by permuting the group labels.
    rng = np.random.default_rng(seed)
    projects = df["project"].values
    uniq = df["project"].unique()
    perm = {p: i for i, p in enumerate(rng.permutation(uniq))}
    pseudo_groups = np.array([perm[p] for p in projects])
    X = np.zeros(len(df))
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, df.label.values, pseudo_groups)):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        # sanity: assert disjoint projects
        assert not (set(tr.project) & set(te.project)), "PROJECT LEAK in fold!"
        log.info("Fold %d/%d: train=%d (%d proj) test=%d (%d proj) test_vuln=%.2f%%",
                 fold + 1, n_splits, len(tr), tr.project.nunique(),
                 len(te), te.project.nunique(), 100 * te.label.mean())
        yield tr.reset_index(drop=True), te.reset_index(drop=True)


def make_imbalanced_test(df: pd.DataFrame, ratio: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Down-sample positives in a test frame to a target vuln fraction (e.g.
    0.05 for 95:5) WITHOUT touching negatives. Use to build the deployment-
    realism test set from a balanced pool."""
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    target_pos = int(ratio / (1 - ratio) * len(neg))
    rng = np.random.default_rng(seed)
    pos_s = pos.sample(n=min(target_pos, len(pos)), random_state=int(rng.integers(1 << 31)))
    return pd.concat([pos_s, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
