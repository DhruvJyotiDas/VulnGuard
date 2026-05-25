"""Dataset loaders -> a single unified schema.

Unified record (pandas DataFrame columns):
    func    : str   -- the C/C++ function source
    label   : int   -- 1 = vulnerable, 0 = safe
    project : str   -- project / repo identifier (REQUIRED for cross-project CV)
    cwe     : str   -- CWE id like "CWE-120" or "" if unknown

Why a unified schema: the reviewers want (a) a second dataset and (b) a
cross-project split. Both are trivial only if every dataset exposes a stable
`project` field. BigVul has it; Devign does NOT natively expose per-function
project, so we derive it (see load_devign). Read those notes before trusting
cross-project numbers on Devign.

NONE of these loaders download data. Point them at files you fetched yourself:
  BigVul : https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset
  Devign : https://github.com/epicosy/devign  (or the HF mirror)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.common import get_logger

log = get_logger(__name__)

REQUIRED_COLS = ["func", "label", "project", "cwe"]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Loader produced a frame missing {missing}; got {list(df.columns)}")
    df = df[REQUIRED_COLS].copy()
    df["func"] = df["func"].astype(str)
    df["label"] = df["label"].astype(int)
    df["project"] = df["project"].astype(str).replace("", "UNKNOWN")
    df["cwe"] = df["cwe"].fillna("").astype(str)
    df = df.dropna(subset=["func"])
    df = df[df["func"].str.len() > 0].reset_index(drop=True)
    pos = int(df["label"].sum())
    log.info("Loaded %d functions | %d vuln (%.2f%%) | %d projects",
             len(df), pos, 100 * pos / max(len(df), 1), df["project"].nunique())
    return df


def load_bigvul(csv_path: str) -> pd.DataFrame:
    """BigVul MSR_20 CSV. Column names in the public release have drifted across
    mirrors; we map the common variants. If your copy differs, fix the mapping
    here rather than downstream."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"BigVul not found at {csv_path}. Download from the ZeoVan repo and "
            f"point --bigvul at the CSV.")
    df = pd.read_csv(csv_path, low_memory=False)

    code_col = _first_present(df, ["func_before", "processed_func", "func", "code"])
    label_col = _first_present(df, ["vul", "target", "label", "is_vul"])
    proj_col = _first_present(df, ["project", "repo", "Project"])
    cwe_col = _first_present(df, ["CWE ID", "cwe", "CWE", "cwe_id"])

    out = pd.DataFrame({
        "func": df[code_col],
        "label": df[label_col],
        # BigVul has no clean per-function project in every mirror; fall back to
        # commit/file hash so cross-project grouping at least never leaks
        # within a group. Prefer a real project column when present.
        "project": df[proj_col] if proj_col else _derive_project(df),
        "cwe": df[cwe_col] if cwe_col else "",
    })
    return _finalize(out)


def load_devign(json_or_csv_path: str) -> pd.DataFrame:
    """Devign (FFmpeg + Qemu). The canonical release does not tag per-function
    project, but it is a 2-project dataset, so we infer project from provenance
    if a `project`/`repo` field exists, else we mark UNKNOWN and you should NOT
    run cross-project CV on it (you'd only have 2 groups -> useless folds)."""
    if json_or_csv_path.endswith(".json"):
        df = pd.read_json(json_or_csv_path)
    else:
        df = pd.read_csv(json_or_csv_path, low_memory=False)
    code_col = _first_present(df, ["func", "code", "function"])
    label_col = _first_present(df, ["target", "label", "vul"])
    proj_col = _first_present(df, ["project", "repo"])
    out = pd.DataFrame({
        "func": df[code_col],
        "label": df[label_col],
        "project": df[proj_col] if proj_col else "UNKNOWN",
        "cwe": "",  # Devign is not CWE-labelled
    })
    if not proj_col:
        log.warning("Devign has no project column -> cross-project CV is INVALID "
                    "on this dataset (2 groups at most). Use it only for the "
                    "second-dataset generalization check, not cross-project.")
    return _finalize(out)


def load_synthetic(n: int = 400, vuln_ratio: float = 0.0574, n_projects: int = 12,
                   seed: int = 0) -> pd.DataFrame:
    """Tiny fake dataset that mimics BigVul's imbalance and project structure.
    Lets you run the WHOLE pipeline (train/eval/adversarial/cross-project) in
    seconds on CPU to catch plumbing bugs before touching the MI300X.

    Vulnerable functions deliberately contain an unsafe sink (strcpy/gets/etc.)
    so even a tiny model learns *something* and the smoke metrics are non-zero;
    safe functions contain bounds checks."""
    rng = np.random.default_rng(seed)
    sinks = ["strcpy(dst, src);", "gets(buf);", "sprintf(buf, fmt, x);", "memcpy(d, s, n);"]
    cwes = ["CWE-120", "CWE-119", "CWE-787", "CWE-125"]
    rows = []
    for i in range(n):
        proj = f"proj_{i % n_projects}"
        is_vuln = int(rng.random() < vuln_ratio)
        if is_vuln:
            body = f"  char buf[10];\n  {rng.choice(sinks)}\n  return 0;"
            cwe = str(rng.choice(cwes))
        else:
            body = "  char buf[64];\n  if (len < 64) { memcpy(buf, src, len); }\n  return 0;"
            cwe = ""
        rows.append({
            "func": f"int func_{i}(char *src, int len) {{\n{body}\n}}",
            "label": is_vuln, "project": proj, "cwe": cwe})
    # guarantee at least a few positives even with tiny n
    if sum(r["label"] for r in rows) < 4:
        for r in rows[:4]:
            r["label"] = 1
            r["func"] = r["func"].replace("if (len < 64)", "").replace("memcpy(buf, src, len);", "strcpy(buf, src);")
            r["cwe"] = "CWE-120"
    return _finalize(pd.DataFrame(rows))


def load_dataset(name: str, path: Optional[str] = None, **kw) -> pd.DataFrame:
    name = name.lower()
    if name == "bigvul":
        return load_bigvul(path)
    if name == "devign":
        return load_devign(path)
    if name == "synthetic":
        return load_synthetic(**kw)
    raise ValueError(f"Unknown dataset '{name}'")


def _first_present(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _derive_project(df: pd.DataFrame) -> pd.Series:
    for c in ["commit_id", "commit", "file_name", "hash"]:
        if c in df.columns:
            return df[c].astype(str)
    return pd.Series(["UNKNOWN"] * len(df))
