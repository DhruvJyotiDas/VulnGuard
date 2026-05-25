"""Shared training loop used by every experiment.

Design choices tied to the review:
  * bf16 autocast by default (MI300X-native; avoids the fp16+30x collapse).
  * `rebalance_each_epoch`: re-undersample safe class each epoch (paper's claim).
  * loss + balancing are the ONLY knobs that vary between baseline and
    VulnGuard, so the ablation is clean.
  * predict() returns raw logits AND probs so eval can run logit_diagnostics
    and calibrate thresholds — the machinery for diagnosing the 0.004 bug.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..data.preprocess import hard_balance
from ..models.factory import CodeDataset
from ..models.losses import build_loss
from ..utils.common import autocast_dtype, get_device, get_logger

log = get_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    batch_size: int = 32          # MI300X has 192GB; no grad-accum needed
    grad_accum: int = 1
    max_len: int = 512
    loss: str = "weighted_ce"     # 'weighted_ce' | 'focal' | 'ce'
    loss_kwargs: dict = field(default_factory=lambda: {"w0": 1.0, "w1": 30.0})
    balance: bool = True          # hard-balance training data
    rebalance_each_epoch: bool = True
    prefer_bf16: bool = True
    num_workers: int = 4
    seed: int = 0


class Trainer:
    def __init__(self, built, cfg: TrainConfig):
        self.built = built
        self.model = built.model
        self.tok = built.tokenizer
        self.cfg = cfg
        self.device = get_device()
        self.amp_dtype = autocast_dtype(cfg.prefer_bf16)
        self.model.to(self.device)
        self.loss_fn = build_loss(cfg.loss, **cfg.loss_kwargs).to(self.device)

    def _loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = CodeDataset(df, self.tok, self.cfg.max_len)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle,
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        cfg = self.cfg
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        # estimate steps from balanced size if balancing
        approx = hard_balance(train_df, seed=cfg.seed) if cfg.balance else train_df
        steps_per_epoch = max(1, len(approx) // (cfg.batch_size * cfg.grad_accum))
        total = steps_per_epoch * cfg.epochs
        sched = get_cosine_schedule_with_warmup(opt, cfg.warmup_steps, total)

        for epoch in range(cfg.epochs):
            if cfg.balance:
                seed = cfg.seed + (epoch if cfg.rebalance_each_epoch else 0)
                ep_df = hard_balance(train_df, seed=seed)
            else:
                ep_df = train_df
            loader = self._loader(ep_df, shuffle=True)
            self.model.train()
            running, opt_steps = 0.0, 0
            opt.zero_grad()
            for step, batch in enumerate(loader):
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                ctx = torch.autocast(device_type="cuda", dtype=self.amp_dtype) \
                    if self.amp_dtype else torch.autocast(device_type="cpu", enabled=False)
                with ctx:
                    logits = self.model(input_ids=ids, attention_mask=mask).logits
                loss = self.loss_fn(logits, y) / cfg.grad_accum
                loss.backward()
                running += loss.item() * cfg.grad_accum
                if (step + 1) % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
                    opt_steps += 1
            msg = f"epoch {epoch+1}/{cfg.epochs} loss={running/max(1,len(loader)):.4f}"
            if val_df is not None:
                m = self.evaluate(val_df, threshold=0.5)
                msg += f" | val_f1@0.5={m['f1']:.3f} val_recall={m['recall']:.3f} pred_pos={m['pred_pos_rate']:.3f}"
                if m["collapsed"]:
                    msg += "  <<< COLLAPSED"
            log.info(msg)
        return self

    @torch.no_grad()
    def predict(self, df: pd.DataFrame):
        """Return (labels, probs[:,1], logits[:,2])."""
        self.model.eval()
        loader = self._loader(df, shuffle=False)
        all_logits, all_y = [], []
        for batch in loader:
            ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            ctx = torch.autocast(device_type="cuda", dtype=self.amp_dtype) \
                if self.amp_dtype else torch.autocast(device_type="cpu", enabled=False)
            with ctx:
                logits = self.model(input_ids=ids, attention_mask=mask).logits
            all_logits.append(logits.float().cpu().numpy())
            all_y.append(batch["labels"].numpy())
        logits = np.concatenate(all_logits)
        labels = np.concatenate(all_y)
        probs = _softmax(logits)[:, 1]
        return labels, probs, logits

    def evaluate(self, df: pd.DataFrame, threshold: float = 0.5):
        from ..eval.metrics import compute_metrics
        labels, probs, _ = self.predict(df)
        return compute_metrics(labels, probs, threshold)


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
