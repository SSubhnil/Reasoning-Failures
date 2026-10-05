from __future__ import annotations

from typing import Callable

import torch

BIPP_LAYERS = [16, 20, 24, 28]
BIPP_HIDDEN = 2560
N_LOGIT_FEATURES = 5
_BLOCK = len(BIPP_LAYERS) * BIPP_HIDDEN

BIPP_LASTTOK_LAYERS = [23, 24]
_LT_BLOCK = len(BIPP_LASTTOK_LAYERS) * BIPP_HIDDEN

def bipp_d_input(n_logit: int = N_LOGIT_FEATURES) -> int:
    return 3 * _BLOCK + n_logit

def bipp_assemble_input(shard: dict, i: int) -> torch.Tensor:
    span = shard["span_feats"][i].float()
    final = shard["final_feats"][i].float()
    ans = shard["answer_feats"][i].float()
    logit = shard["logit_features"][i].float()
    L = span.shape[0]
    if final.shape[0] != L or logit.shape[0] != L:
        raise ValueError(
            f"[bipp] step-count mismatch i={i}: span{tuple(span.shape)} "
            f"final{tuple(final.shape)} logit{tuple(logit.shape)}"
        )
    ans_b = ans.unsqueeze(0).expand(L, -1)
    return torch.cat([span, final, ans_b, logit], dim=-1)

def bipp_d_input_lasttok() -> int:
    return 2 * _LT_BLOCK

def bipp_assemble_input_lasttok(shard: dict, i: int) -> torch.Tensor:
    final = shard["features"][i].float()
    if final.shape[-1] != _LT_BLOCK:
        raise ValueError(
            f"[bipp] lasttok record i={i} has width {final.shape[-1]}, expected {_LT_BLOCK}"
        )
    L = final.shape[0]
    ans_b = final[-1].unsqueeze(0).expand(L, -1)
    return torch.cat([final, ans_b], dim=-1)

BIPP_MODES: dict[str, tuple[Callable[[dict, int], torch.Tensor], Callable[[], int], tuple[str, ...]]] = {
    "span": (bipp_assemble_input, bipp_d_input,
             ("span_feats", "final_feats", "answer_feats", "logit_features")),
    "lasttok": (bipp_assemble_input_lasttok, bipp_d_input_lasttok, ("features",)),
}
