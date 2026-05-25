"""Explainability that goes beyond attention — the reviewers' explicit ask.

Three methods, increasing strength of evidence:
  1. Integrated Gradients (captum) over input embeddings -> token attributions.
  2. Occlusion: mask each token, measure drop in P(vuln) -> token importance.
  3. Counterfactual: delete the suspected sink (e.g. strcpy) and check whether
     P(vuln) actually flips. This is the strongest single piece of evidence that
     the model keys on the vulnerability and not surrounding noise.

Reporting guidance for the rejoinder: don't claim 'attention proves semantic
learning'. Claim 'IG + occlusion + counterfactual converge on the security
sink for N functions' and report aggregate counterfactual flip-rate.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from ..utils.common import get_device


@torch.no_grad()
def _prob_vuln(model, tok, code: str, device, max_len=512) -> float:
    enc = tok(code, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    logits = model(**enc).logits.float()
    return torch.softmax(logits, -1)[0, 1].item()


def occlusion_attributions(model, tok, code: str, max_len=512) -> List[Tuple[str, float]]:
    """Importance of each token = P(vuln|full) - P(vuln|token masked)."""
    device = get_device(); model.eval()
    enc = tok(code, truncation=True, max_length=max_len, return_tensors="pt")
    ids = enc["input_ids"][0]
    base = _prob_vuln(model, tok, code, device, max_len)
    mask_id = tok.mask_token_id if tok.mask_token_id is not None else tok.unk_token_id
    out = []
    for i in range(1, len(ids) - 1):  # skip special tokens
        masked = ids.clone(); masked[i] = mask_id
        with torch.no_grad():
            logits = model(input_ids=masked.unsqueeze(0).to(device)).logits.float()
        p = torch.softmax(logits, -1)[0, 1].item()
        out.append((tok.convert_ids_to_tokens(ids[i].item()), base - p))
    return sorted(out, key=lambda x: -abs(x[1]))


def integrated_gradients(model, tok, code: str, steps: int = 32, max_len=512):
    """IG over the embedding layer. Returns [(token, attribution)]."""
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        raise ImportError("pip install captum to use integrated_gradients")
    device = get_device(); model.eval()
    enc = tok(code, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    ids = enc["input_ids"]

    emb_layer = model.get_input_embeddings()

    def fwd(input_ids):
        return torch.softmax(model(input_ids=input_ids).logits, -1)[:, 1]

    lig = LayerIntegratedGradients(fwd, emb_layer)
    ref = torch.full_like(ids, tok.pad_token_id)
    attrs = lig.attribute(ids, baselines=ref, n_steps=steps)
    attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    toks = tok.convert_ids_to_tokens(ids[0])
    return sorted(zip(toks, attrs), key=lambda x: -abs(x[1]))


def counterfactual_sink_removal(model, tok, code: str, sinks=None, max_len=512):
    """Remove each candidate sink call and measure P(vuln) drop. Returns a list
    of (sink, p_before, p_after, flipped)."""
    sinks = sinks or ["strcpy", "gets", "sprintf", "memcpy", "strcat", "scanf"]
    device = get_device(); model.eval()
    p_before = _prob_vuln(model, tok, code, device, max_len)
    results = []
    for s in sinks:
        if s in code:
            # neutralize the call into a safe no-op comment
            cf = code.replace(s, f"/*{s}*/safe_noop")
            p_after = _prob_vuln(model, tok, cf, device, max_len)
            results.append({
                "sink": s, "p_before": p_before, "p_after": p_after,
                "delta": p_before - p_after,
                "flipped": bool(p_before >= 0.5 > p_after),
            })
    return results
