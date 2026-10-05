from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

def _probe_loss_confidence(
    probe, device, hidden_states, is_correct, length_mask,
    *,
    loss: str = "bce",
    label_smooth: float = 0.0,
):

    if loss not in ("bce", "brier"):
        raise ValueError(f"[Probe] unknown confidence loss={loss!r}; expected 'bce' | 'brier'.")
    if not torch.isfinite(hidden_states).all():
        raise RuntimeError("[Probe] hidden_states contains NaN/Inf")

    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if hidden_states.is_cuda else contextlib.nullcontext())
    with amp:
        conf_pred = probe(hidden_states, length_mask=length_mask)
        known = is_correct >= 0
        if length_mask.any() and known.any():
            B, L = conf_pred.shape
            tgt = is_correct.unsqueeze(1).expand(B, L)
            combined = length_mask & known.unsqueeze(1).expand(B, L)
            if combined.any():
                p = conf_pred[combined].float().clamp(1e-6, 1.0 - 1e-6)
                y = tgt[combined].float()
                if label_smooth > 0.0:
                    y = y * (1.0 - label_smooth) + 0.5 * label_smooth
                if loss == "bce":
                    l = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).mean()
                else:
                    l = (p - y).pow(2).mean()
            else:
                l = torch.zeros((), device=device)
        else:
            l = torch.zeros((), device=device)

    return l.to(device), {"loss_conf": float(l.detach())}

def _advantage_from_q(q_values: list[float], baseline: str = "trace_mean") -> list[float]:
    if not q_values:
        return []
    if baseline == "prev":
        return [0.0] + [q_values[t] - q_values[t - 1] for t in range(1, len(q_values))]
    if baseline == "trace_mean":
        m = sum(q_values) / len(q_values)
        return [q - m for q in q_values]
    raise ValueError(f"[Probe] unknown advantage baseline={baseline!r}; expected 'trace_mean' | 'prev'.")

def compute_blindspot_confidence(
    q_values: list[float],
    is_correct: bool,
    blindspot_threshold: float = 0.0,
    pit_signal: str = "raw_q",
    advantage_baseline: str = "trace_mean",
) -> dict:

    if not q_values:
        return {
            "trace_confidence": 0.0,
            "step_confidences": [],
            "pit_step": -1,
            "pit_confidence": 0.0,
            "is_blindspot": False,
        }

    q_min = min(q_values)
    q_max = max(q_values)
    q_range = q_max - q_min
    eps = 1e-8

    trace_confidence = sum(q_values) / len(q_values)
    step_confidences = [(q - q_min) / (q_range + eps) for q in q_values]

    if pit_signal == "advantage":
        adv = _advantage_from_q(q_values, baseline=advantage_baseline)
        pit_step = int(adv.index(min(adv)))
    elif pit_signal == "raw_q":
        pit_step = int(q_values.index(q_min))
    else:
        raise ValueError(f"[Probe] unknown pit_signal={pit_signal!r}; expected 'raw_q' | 'advantage'.")
    pit_confidence = step_confidences[pit_step]
    is_blindspot = (not is_correct) and (trace_confidence > blindspot_threshold)

    return {
        "trace_confidence": trace_confidence,
        "step_confidences": step_confidences,
        "pit_step": pit_step,
        "pit_confidence": pit_confidence,
        "is_blindspot": is_blindspot,
    }

def first_passage_pit(curve, theta: float = 0.1) -> int:
    best_t, best_v = -1, float("inf")
    for t, c in enumerate(curve):
        if c <= theta:
            return t
        if c < best_v:
            best_t, best_v = t, c
    return best_t

def binomial_nll(
    value: torch.Tensor,
    target: torch.Tensor,
    n_rollouts: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:

    v = value.float().clamp(eps, 1.0 - eps)
    p = target.float().clamp(0.0, 1.0)
    k = n_rollouts.float().clamp(min=eps)
    bce = -(p * torch.log(v) + (1.0 - p) * torch.log(1.0 - v))
    return (k * bce).sum() / k.sum().clamp(min=eps)

def _probe_loss_value(
    probe, device, hidden_states, targets, target_mask, n_rollouts, length_mask,
):

    if not torch.isfinite(hidden_states).all():
        raise RuntimeError("[value] hidden_states contains NaN/Inf")
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if hidden_states.is_cuda else contextlib.nullcontext())
    with amp:
        value = probe(hidden_states, length_mask=length_mask)
        m = target_mask & length_mask
        if m.any():
            loss = binomial_nll(value[m], targets[m], n_rollouts[m])
        else:
            loss = torch.zeros((), device=device)
    return loss, {"loss_value": float(loss.detach()), "n_anchored": int(m.sum())}

def _probe_loss_pointer(
    probe, device, hidden_states, is_correct, length_mask,
    *,
    gold_pit: torch.Tensor | None = None,
    regime: torch.Tensor | None = None,
    anchor_targets: torch.Tensor | None = None,
    anchor_mask: torch.Tensor | None = None,
    anchor_weights: torch.Tensor | None = None,
    branch: dict | None = None,
    lambda_ptr: float = 1.0,
    lambda_regime: float = 1.0,
    lambda_env: float = 0.25,
    lambda_branch: float = 0.25,
    beta_conf: float = 0.5,
    regime_class_weights: torch.Tensor | None = None,
    smooth_sigma: float = 1.0,
    branch_margin: float = 1.0,
):

    if not torch.isfinite(hidden_states).all():
        raise RuntimeError("[Probe] hidden_states contains NaN/Inf")

    B, L, _ = hidden_states.shape
    zero = torch.zeros((), device=device)
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if hidden_states.is_cuda else contextlib.nullcontext())

    with amp:
        pointer_logits, regime_logits, R_hat, conf_pred = probe(
            hidden_states, length_mask=length_mask, return_pointer=True,
        )
        known = is_correct >= 0

        if gold_pit is not None:

            masked_logits = pointer_logits.float().masked_fill(~length_mask, float("-inf"))
            logprob = torch.log_softmax(masked_logits, dim=1)
            step_idx = torch.arange(L, device=device).float()
            terms = []
            for b in range(B):
                p = int(gold_pit[b])
                if p < 0:
                    continue
                valid = length_mask[b]
                tgt = torch.exp(-0.5 * ((step_idx - float(p)) / smooth_sigma) ** 2)
                tgt = tgt.masked_fill(~valid, 0.0)
                denom = tgt.sum()
                if float(denom) <= 0.0:
                    continue
                tgt = tgt / denom

                terms.append(-(tgt[valid] * logprob[b][valid]).sum())
            loss_ptr = torch.stack(terms).mean() if terms else zero
        else:
            loss_ptr = zero

        if regime is not None:
            rk = regime >= 0
            if rk.any():
                loss_regime = F.cross_entropy(
                    regime_logits[rk], regime[rk], weight=regime_class_weights,
                )
            else:
                loss_regime = zero
        else:
            loss_regime = zero

        if anchor_targets is not None and anchor_mask is not None:
            m = anchor_mask & length_mask
            if m.any():
                r = R_hat[m].clamp(1e-6, 1.0 - 1e-6)
                v = anchor_targets[m].clamp(0.0, 1.0)
                bce = -(v * torch.log(r) + (1.0 - v) * torch.log(1.0 - r))
                if anchor_weights is not None:
                    w = anchor_weights[m].clamp(min=0.0)
                    loss_env = (w * bce).sum() / w.sum().clamp(min=1e-6)
                else:
                    loss_env = bce.mean()
            else:
                loss_env = zero
        else:
            loss_env = zero

        if branch is not None:
            pos_feats = branch["pos_feats"].to(device)
            pos_mask = branch["pos_mask"].to(device)
            pos_t = branch["pos_t"].to(device)
            neg_b = branch["neg_b"].to(device)
            neg_t = branch["neg_t"].to(device)
            if not torch.isfinite(pos_feats).all():
                raise RuntimeError("[Probe] branch pos_feats contains NaN/Inf")
            s_pos, _, _, _ = probe(
                pos_feats, length_mask=pos_mask, return_pointer=True,
            )
            pr = torch.arange(pos_t.shape[0], device=device)
            s_plus = s_pos[pr, pos_t]
            s_minus = pointer_logits[neg_b, neg_t]

            loss_branch = F.softplus(s_plus - s_minus + branch_margin).mean()
        else:
            loss_branch = zero

        if length_mask.any() and known.any():
            tgt = is_correct.unsqueeze(1).expand(B, L)
            combined = length_mask & known.unsqueeze(1).expand(B, L)
            loss_brier = ((conf_pred[combined] - tgt[combined]).pow(2).mean()
                          if combined.any() else zero)
        else:
            loss_brier = zero

        total_loss = (lambda_ptr * loss_ptr
                      + lambda_regime * loss_regime
                      + lambda_env * loss_env
                      + lambda_branch * loss_branch
                      + beta_conf * loss_brier)

    stats = {
        "loss_ptr": float(loss_ptr),
        "loss_regime": float(loss_regime),
        "loss_env": float(loss_env),
        "loss_branch": float(loss_branch),
        "loss_brier": float(loss_brier),
    }
    return total_loss.to(device), stats
