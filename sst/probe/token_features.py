from __future__ import annotations

import torch

LOGIT_FEATURE_NAMES: tuple[str, ...] = (
    "entropy",
    "varentropy",
    "max_prob",
    "prob_margin",
    "top5_mass",
)
N_LOGIT_FEATURES: int = len(LOGIT_FEATURE_NAMES)

def decision_token_features(logits: torch.Tensor, *, top_k: int = 5) -> torch.Tensor:
    if logits.shape[-1] < top_k:
        raise ValueError(
            f"[token_features] vocab dim {logits.shape[-1]} < top_k={top_k}."
        )
    x = logits.float()
    logp = x - torch.logsumexp(x, dim=-1, keepdim=True)
    p = logp.exp()

    neg_h = (p * logp).sum(dim=-1)
    entropy = -neg_h

    e_logp_sq = (p * logp.pow(2)).sum(dim=-1)
    varentropy = (e_logp_sq - neg_h.pow(2)).clamp_min(0.0)

    topk_p = p.topk(top_k, dim=-1).values
    max_prob = topk_p[..., 0]
    prob_margin = topk_p[..., 0] - topk_p[..., 1]
    top5_mass = topk_p.sum(dim=-1)

    return torch.stack(
        [entropy, varentropy, max_prob, prob_margin, top5_mass], dim=-1,
    )

def concat_hidden_and_logit(
    hidden: torch.Tensor, logit_feats: torch.Tensor,
) -> torch.Tensor:

    if hidden.shape[:2] != logit_feats.shape[:2]:
        raise ValueError(
            f"[token_features] step-axis mismatch: hidden {tuple(hidden.shape[:2])} "
            f"vs logit {tuple(logit_feats.shape[:2])}."
        )
    return torch.cat([hidden, logit_feats.to(hidden.dtype)], dim=-1)
