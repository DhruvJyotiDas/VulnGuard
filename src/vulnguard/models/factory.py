"""Model factory + tokenized torch Dataset.

Every model is a HF AutoModelForSequenceClassification head on a code/text
encoder, so the trainer is shared. The point of the *fair-baseline* fix is that
CodeBERT and VulnGuard differ ONLY in (balancing, loss weight) — same backbone,
same trainer — so any delta is attributable to those two knobs, not to setup
differences. That is the ablation Reviewer 1 wants.

Faithful-proxy honesty:
  - microsoft/codebert-base, microsoft/graphcodebert-base : exact.
  - LineVul : the paper is line-level; here we use its RoBERTa backbone for
    FUNCTION-level classification, which is the standard function-granularity
    comparison. Not line-level. Flagged.
  - VulBERTa : claudios/VulBERTa-MLP exists on HF; if unavailable in your env,
    it falls back to roberta-base and prints a loud warning.
  - VulDeBERT : no clean public checkpoint matches the paper. We fine-tune
    microsoft/deberta-v3-base as the closest honest proxy. Flagged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils.common import get_logger

log = get_logger(__name__)

MODEL_REGISTRY = {
    "codebert": "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
    "vulnguard": "microsoft/codebert-base",   # same backbone as codebert ON PURPOSE
    "linevul": "microsoft/codebert-base",      # LineVul uses a CodeBERT/RoBERTa backbone
    "vulberta": "claudios/VulBERTa-MLP",
    "vuldebert": "microsoft/deberta-v3-base",  # PROXY, see module docstring
}

PROXY_MODELS = {"linevul", "vulberta", "vuldebert"}


@dataclass
class BuiltModel:
    model: torch.nn.Module
    tokenizer: object
    name: str
    is_proxy: bool


def build_model(name: str, num_labels: int = 2):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Known: {list(MODEL_REGISTRY)}")
    hf_id = MODEL_REGISTRY[name]
    if name in PROXY_MODELS:
        log.warning("Model '%s' is a FAITHFUL PROXY via '%s', not the original "
                    "release. Disclose this in the rejoinder.", name, hf_id)
    try:
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=num_labels)
    except Exception as e:  # pragma: no cover - network/availability dependent
        if name == "vulberta":
            log.error("Could not load VulBERTa (%s). Falling back to roberta-base. "
                      "This is NOT VulBERTa — fix before reporting.", e)
            tok = AutoTokenizer.from_pretrained("roberta-base")
            model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)
        else:
            raise
    return BuiltModel(model=model, tokenizer=tok, name=name, is_proxy=name in PROXY_MODELS)


class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len: int = 512):
        self.texts = df["func"].tolist()
        self.labels = df["label"].tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }
