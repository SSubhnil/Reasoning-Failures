import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

class BaseModel(nn.Module):
    def __init__(self, hf_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_name, dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.gen_kwargs = {
            "do_sample": True,
            "temperature": 0.7,
            "max_new_tokens": 256,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 6,
        }
