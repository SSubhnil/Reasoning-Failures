from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

class PretrainedTranscoder(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.d_model = state["b_dec"].shape[0]
        self.d_transcoder = state["b_enc"].shape[0]

        self.enc = nn.Linear(self.d_model, self.d_transcoder, bias=True)
        self.enc.weight = nn.Parameter(state["W_enc"].to(dtype), requires_grad=False)
        self.enc.bias = nn.Parameter(state["b_enc"].to(dtype), requires_grad=False)

        self.dec = nn.Linear(self.d_transcoder, self.d_model, bias=True)
        self.dec.weight = nn.Parameter(state["W_dec"].T.contiguous().to(dtype), requires_grad=False)
        self.dec.bias = nn.Parameter(state["b_dec"].to(dtype), requires_grad=False)

        if "activation_function.threshold" in state:
            self.register_buffer("_threshold", state["activation_function.threshold"].to(dtype))
        else:
            self._threshold = None

        for p in self.parameters():
            p.requires_grad = False

def load_pretrained_transcoders(
    repo_id: str,
    target_layers: list[int],
    dtype: torch.dtype = torch.bfloat16,
) -> nn.ModuleDict:

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    block_indices = [L - 1 for L in target_layers]
    local_dir = Path(repo_id).expanduser()
    use_local_dir = local_dir.is_dir()

    def _resolve_layer_path(block_idx: int) -> Path:
        if use_local_dir:
            path = local_dir / f"layer_{block_idx}.safetensors"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing pretrained transcoder weights: {path}"
                )
            return path
        return Path(hf_hub_download(repo_id, filename=f"layer_{block_idx}.safetensors"))

    if dist.is_initialized() and not use_local_dir:
        if dist.get_rank() == 0:
            for bi in block_indices:
                _resolve_layer_path(bi)
        dist.barrier()

    transcoders = nn.ModuleDict()
    for L, bi in zip(target_layers, block_indices):
        path = _resolve_layer_path(bi)
        state = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        transcoders[str(L)] = PretrainedTranscoder(state, dtype=dtype)
    return transcoders
