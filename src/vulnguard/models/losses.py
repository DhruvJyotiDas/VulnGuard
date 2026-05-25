"""Losses. Weighted CE is the paper's method; focal is offered as a stronger
alternative the rejoinder can compare against (reviewers like seeing that the
chosen knob beats obvious alternatives)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCE(nn.Module):
    """CE with class weight [w0, w1]. w1=30 reproduces the paper's 30x penalty
    on false negatives. Computed in fp32 internally for stability under large
    weights — this matters: doing it in fp16 is a plausible cause of the
    baseline's 0.004 collapse."""

    def __init__(self, w0: float = 1.0, w1: float = 30.0):
        super().__init__()
        self.register_buffer("weight", torch.tensor([w0, w1], dtype=torch.float32))

    def forward(self, logits, target):
        return F.cross_entropy(logits.float(), target, weight=self.weight.to(logits.device))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.95):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight on the positive (vulnerable) class

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        w = torch.tensor([1 - self.alpha, self.alpha], device=logits.device)
        focal = -w[target] * (1 - p[torch.arange(len(target)), target]) ** self.gamma \
                * logp[torch.arange(len(target)), target]
        return focal.mean()


def build_loss(kind: str = "weighted_ce", **kw):
    kind = kind.lower()
    if kind in ("weighted_ce", "wce"):
        return WeightedCE(w0=kw.get("w0", 1.0), w1=kw.get("w1", 30.0))
    if kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha", 0.95))
    if kind in ("ce", "standard"):
        return WeightedCE(w0=1.0, w1=1.0)  # unweighted
    raise ValueError(f"Unknown loss '{kind}'")
