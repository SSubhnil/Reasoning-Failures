from __future__ import annotations

from sst.student import base
from sst.shared.schemas import GenerationOutput
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from sst.probe.sae import load_pretrained_transcoders
import threading

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sst.inference.vllm_client import VLLMClient

from sst.shared.rescue import FORCE_FINAL_SUFFIX

_TOKENIZER_LOCK = threading.RLock()

class StudentModel(base.BaseModel):
    def __init__(self, hf_name,
                 step_extractor,
                 system_prompt: str = (
                     "You are a student learning math."
                     "Think and reason step-wise to give a final answer in \\boxed{}."),
                 pretrained_transcoder_repo: str | None = None,
                 target_layers: list[int] | None = None,
                 vllm_client: VLLMClient | None = None,):
        super().__init__(hf_name)
        self.step_extractor = step_extractor
        self.system_prompt = system_prompt
        self.vllm_client = vllm_client

        self.model.config.output_hidden_states = False

        self.n_layers = self.model.config.num_hidden_layers
        self.middle_layer = self.n_layers //2
        if target_layers:
            self.target_layers = sorted(target_layers)
        else:
            self.target_layers = [
                self.middle_layer + 1,
                self.middle_layer + 2,
                self.middle_layer + 3,
            ]

        self.hidden_size = self.model.config.hidden_size

        self.pretrained_transcoder_repo = pretrained_transcoder_repo
        self.transcoders = (load_pretrained_transcoders(pretrained_transcoder_repo, self.target_layers)
                            if pretrained_transcoder_repo else nn.ModuleDict())

    def _wrap_prompt(self, prompt: List[str]) -> List[str]:
        with _TOKENIZER_LOCK:
            return [self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": p}],
                        tokenize=False, add_generation_prompt=True)
                    for p in prompt]

    def _prepare_step_indices(self, step_spans: List[List[Tuple[int, int]]], max_seq_len: int) -> Tuple[torch.LongTensor, List[int]]:
        batch_size = len(step_spans)
        max_steps = max(len(s) for s in step_spans) if step_spans else 0

        if max_steps == 0:
            return torch.zeros((batch_size, 0), dtype=torch.long, device=self.model.device), [0]*batch_size

        indices = torch.zeros((batch_size, max_steps), dtype=torch.long, device=self.model.device)
        valid_counts = []

        for b, spans in enumerate(step_spans):
            valid_counts.append(len(spans))
            for i, (start, end) in enumerate(spans):

                idx = min(end - 1, max_seq_len - 1)
                indices[b, i] = idx

        return indices, valid_counts

    @torch.no_grad()
    def generate(self, prompt: List[str], num_traces: int = 1, **override_kwargs) -> GenerationOutput:
        prompt_list = self._wrap_prompt(prompt)
        device_type = next(iter(self.model.parameters())).device.type
        force_final_on_truncation = override_kwargs.pop("force_final_on_truncation", True)
        force_final_max_new_tokens = int(override_kwargs.pop("force_final_max_new_tokens", 32))
        with _TOKENIZER_LOCK:
            _enc = self.tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True,)
        prompt_lens = _enc.to(device_type)["attention_mask"].sum(dim=1).tolist()
        gen_cfg = dict(self.gen_kwargs, **override_kwargs, num_return_sequences=num_traces)
        (all_traces, all_extracted_steps, all_step_spans,
         all_finish_reasons, all_token_counts) = self._generate_via_vllm(
             prompt_list=prompt_list,
             prompt_lens=prompt_lens,
             num_traces_per_prompt=int(gen_cfg["num_return_sequences"]),
             gen_cfg=gen_cfg,
             force_final_on_truncation=force_final_on_truncation,
             force_final_max_new_tokens=force_final_max_new_tokens,
         )
        return GenerationOutput(
            traces=all_traces,
            extracted_steps=all_extracted_steps,
            hidden_states=torch.empty(0),
            transcoder_z={},
            target_states=None,
            step_spans=all_step_spans,
            finish_reasons=all_finish_reasons,
            token_counts=all_token_counts,
        )

    @torch.no_grad()
    def reencode_traces(
        self,
        prompts: list[str],
        traces: list[str],
        step_spans: list[list[tuple[int, int]]],
    ) -> "GenerationOutput":

        full_text = [p + t for p, t in zip(prompts, traces)]

        needed_indices = list(self.target_layers)
        enc = self.tokenizer(full_text, padding=False, truncation=False, return_length=True)
        full_lens = [int(x) for x in enc["length"]] if "length" in enc else [len(ids) for ids in enc["input_ids"]]

        global_max_steps = max((len(s) for s in step_spans), default=0)

        captured_final: Dict[int, torch.Tensor] = {}
        for hs_idx in needed_indices:
            captured_final[hs_idx] = torch.zeros(
                (len(full_text), global_max_steps, self.hidden_size),
                device=self.model.device, dtype=torch.bfloat16,
            )

        buckets = self._make_length_buckets(
            lengths=full_lens,
            bucket_width=512,
            token_budget=64_000,
        )

        for bucket in buckets:
            bucket_text = [full_text[i] for i in bucket]
            bucket_spans = [step_spans[i] for i in bucket]

            toks = self.tokenizer(bucket_text, return_tensors="pt", padding=True).to(self.model.device)
            step_end_indices, _ = self._prepare_step_indices(bucket_spans, toks["input_ids"].size(1))

            captured_bucket, _ = self._collect_selected_hidden_states(
                toks, hidden_state_indices=needed_indices, step_end_indices=step_end_indices,
            )

            bucket_idx_t = torch.tensor(bucket, dtype=torch.long, device=self.model.device)
            for hs_idx in needed_indices:
                x = captured_bucket[hs_idx]
                if x.size(1) < global_max_steps:
                    pad = torch.zeros(x.size(0), global_max_steps - x.size(1), x.size(2),
                                      device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=1)
                captured_final[hs_idx].index_copy_(0, bucket_idx_t, x)

            del toks, captured_bucket, bucket_idx_t

        picked = [captured_final[i] for i in self.target_layers]
        hidden_states = torch.cat(picked, dim=-1) if picked else torch.empty(
            len(traces), 0, 0, device=self.model.device, dtype=torch.bfloat16,
        )

        del captured_final
        torch.cuda.empty_cache()

        return GenerationOutput(
            traces=traces,
            extracted_steps=[[] for _ in traces],
            hidden_states=hidden_states,
            transcoder_z={},
            target_states=None,
            step_spans=step_spans,
            finish_reasons=["reencode" for _ in traces],
        )

    @torch.no_grad()
    def extract_decision_logit_features(
        self,
        prompts: list[str],
        traces: list[str],
        step_spans: list[list[tuple[int, int]]],
        *,
        top_k: int = 5,
        token_budget: int = 24_000,
    ) -> torch.Tensor:

        from sst.probe.token_features import N_LOGIT_FEATURES, decision_token_features

        causal_lm = self.model
        if hasattr(causal_lm, "peft_config"):
            causal_lm = causal_lm.base_model.model
        lm_head = causal_lm.lm_head
        backbone = self._get_base_transformer()

        full_text = [p + t for p, t in zip(prompts, traces)]
        enc = self.tokenizer(full_text, padding=False, truncation=False, return_length=True)
        full_lens = (
            [int(x) for x in enc["length"]]
            if "length" in enc else [len(ids) for ids in enc["input_ids"]]
        )
        global_max_steps = max((len(s) for s in step_spans), default=0)

        out = torch.zeros(
            (len(full_text), global_max_steps, N_LOGIT_FEATURES), dtype=torch.float32,
        )
        if global_max_steps == 0:
            return out

        buckets = self._make_length_buckets(
            lengths=full_lens, bucket_width=512, token_budget=token_budget,
        )
        for bucket in buckets:
            bucket_text = [full_text[i] for i in bucket]
            bucket_spans = [step_spans[i] for i in bucket]
            toks = self.tokenizer(
                bucket_text, return_tensors="pt", padding=True,
            ).to(self.model.device)
            step_end_indices, valid_counts = self._prepare_step_indices(
                bucket_spans, toks["input_ids"].size(1),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = backbone(
                    input_ids=toks["input_ids"],
                    attention_mask=toks["attention_mask"],
                    use_cache=False,
                ).last_hidden_state
                idx = step_end_indices.unsqueeze(-1).expand(-1, -1, hs.size(-1))
                dec_hidden = torch.gather(hs, dim=1, index=idx)
                dec_logits = lm_head(dec_hidden)
                feats = decision_token_features(dec_logits, top_k=top_k)
            feats = feats.cpu()
            for bi, gi in enumerate(bucket):
                s = valid_counts[bi]
                if s:
                    out[gi, :s] = feats[bi, :s]
            del toks, hs, dec_hidden, dec_logits, feats
            torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def extract_span_pooled_features(
        self,
        prompts: list[str],
        traces: list[str],
        step_spans: list[list[tuple[int, int]]],
        *,
        layers: list[int],
    ) -> Dict[str, list[torch.Tensor]]:

        layers = [int(L) for L in layers]
        if not layers:
            raise ValueError("[bipp] layers must be non-empty.")
        if any(L <= 0 for L in layers):
            raise ValueError(f"[bipp] layers must be positive HF indices, got {layers}.")

        full_text = [p + t for p, t in zip(prompts, traces)]
        enc = self.tokenizer(full_text, padding=False, truncation=False, return_length=True)
        full_lens = (
            [int(x) for x in enc["length"]]
            if "length" in enc else [len(ids) for ids in enc["input_ids"]]
        )

        n = len(full_text)
        D = self.hidden_size
        span_mean: list[torch.Tensor | None] = [None] * n
        step_final: list[torch.Tensor | None] = [None] * n
        answer: list[torch.Tensor | None] = [None] * n

        blocks = self._get_decoder_blocks()
        base = self._get_base_transformer()
        needed = sorted(set(layers))
        for L in needed:
            block_idx = L - 1
            if block_idx < 0 or block_idx >= len(blocks):
                raise IndexError(
                    f"[bipp] layer index {L} maps to block_idx={block_idx}, "
                    f"but decoder has {len(blocks)} blocks."
                )
        max_needed_block = max(L - 1 for L in needed)
        exit_block_idx = max_needed_block + 1

        buckets = self._make_length_buckets(
            lengths=full_lens, bucket_width=512, token_budget=64_000,
        )
        for bucket in buckets:
            bucket_text = [full_text[i] for i in bucket]
            toks = self.tokenizer(
                bucket_text, return_tensors="pt", padding=True,
            ).to(self.model.device)
            seq_len = int(toks["input_ids"].size(1))

            captured: Dict[int, torch.Tensor] = {}
            handles = []

            def _hook_factory(hs_idx: int):
                def _hook(_module, _inp, out):
                    hs = out[0] if isinstance(out, (tuple, list)) else out
                    captured[hs_idx] = hs
                return _hook

            for L in needed:
                handles.append(blocks[L - 1].register_forward_hook(_hook_factory(L)))
            if exit_block_idx < len(blocks):
                def _early_exit_hook(_module, _inp, _out):
                    raise StudentModel._EarlyExitSignal()
                handles.append(blocks[exit_block_idx].register_forward_hook(_early_exit_hook))

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _ = base.forward(
                        input_ids=toks["input_ids"],
                        attention_mask=toks["attention_mask"],
                        output_hidden_states=False,
                        use_cache=False,
                    )
            except StudentModel._EarlyExitSignal:
                pass
            finally:
                for h in handles:
                    h.remove()

            for L in needed:
                if L not in captured:
                    raise RuntimeError(f"[bipp] failed to capture hidden state index {L}.")
                cl = captured[L]
                if not torch.isfinite(cl).all():
                    raise RuntimeError(f"[bipp] NaN/Inf in captured layer {L} hidden states.")

            for bi, gi in enumerate(bucket):
                spans = step_spans[gi]
                S = len(spans)
                sm = torch.zeros((S, len(layers) * D), dtype=torch.bfloat16)
                sf = torch.zeros((S, len(layers) * D), dtype=torch.bfloat16)
                for si, (start, end) in enumerate(spans):

                    e = min(int(end), seq_len)
                    s = min(int(start), e - 1)
                    blk_mean, blk_final = [], []
                    for L in layers:
                        hs = captured[L][bi]
                        blk_mean.append(hs[s:e].mean(dim=0))
                        blk_final.append(hs[e - 1])
                    sm[si] = torch.cat(blk_mean, dim=-1).to(torch.bfloat16)
                    sf[si] = torch.cat(blk_final, dim=-1).to(torch.bfloat16)
                span_mean[gi] = sm
                step_final[gi] = sf
                answer[gi] = sf[-1].clone() if S else torch.zeros(len(layers) * D, dtype=torch.bfloat16)

            del toks, captured
            torch.cuda.empty_cache()

        for gi in range(n):
            spans = step_spans[gi]
            for si, (start, end) in enumerate(spans):
                if int(end) - int(start) == 1:
                    if not torch.allclose(
                        span_mean[gi][si].float(), step_final[gi][si].float(),
                        rtol=1e-3, atol=1e-3,
                    ):
                        raise RuntimeError(
                            f"[bipp] single-token step span_mean != step_final "
                            f"(trace {gi}, step {si}) - span/offset bug."
                        )

        return {"span_mean": span_mean, "step_final": step_final, "answer": answer}

    def _generate_via_vllm(
        self,
        prompt_list: List[str],
        prompt_lens: List[int],
        num_traces_per_prompt: int,
        gen_cfg: dict,
        force_final_on_truncation: bool,
        force_final_max_new_tokens: int,
    ) -> Tuple[List[str], List[List[str]], List, List[str], List[int]]:

        assert self.vllm_client is not None

        temperature = float(gen_cfg.get("temperature", 0.7))
        max_tokens = int(gen_cfg.get("max_new_tokens", 2048))
        seed = gen_cfg.get("seed")

        result = self.vllm_client.generate_batch(
            prompt_list,
            n=num_traces_per_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        traces: List[str] = list(result.texts)
        token_counts: List[int] = [int(c) for c in result.token_counts]

        if force_final_on_truncation:
            needs_rescue = [
                b_idx for b_idx, t in enumerate(traces)
                if not self._has_end_think_token(t) and "\\boxed{" not in t
            ]

            if needs_rescue:
                ctx_limit = self._get_context_limit()
                force_suffix = FORCE_FINAL_SUFFIX
                with _TOKENIZER_LOCK:
                    suffix_ids = self.tokenizer(force_suffix, add_special_tokens=False).input_ids
                suffix_len = len(suffix_ids)

                rescue_prompts: List[str] = []
                rescue_trace_tails: List[str] = []
                rescue_map: List[int] = []

                for b_idx in needs_rescue:
                    prompt_idx = b_idx // num_traces_per_prompt
                    orig_prompt_str = prompt_list[prompt_idx]
                    prompt_len_actual = int(prompt_lens[prompt_idx])

                    max_tail = ctx_limit - prompt_len_actual - suffix_len - force_final_max_new_tokens
                    max_tail = max(0, max_tail)

                    t = traces[b_idx]
                    if max_tail == 0:
                        tail_text = ""
                    else:
                        with _TOKENIZER_LOCK:
                            t_ids = self.tokenizer(t, add_special_tokens=False).input_ids
                            if len(t_ids) > max_tail:
                                t_ids = t_ids[-max_tail:]
                            tail_text = self.tokenizer.decode(t_ids, skip_special_tokens=True)

                    rescue_prompt = orig_prompt_str + tail_text.rstrip() + force_suffix
                    rescue_prompts.append(rescue_prompt)
                    rescue_trace_tails.append(tail_text.rstrip())
                    rescue_map.append(b_idx)

                rescue_gens = self.vllm_client.rescue_generate(
                    rescue_prompts, max_tokens=force_final_max_new_tokens,
                )

                for gen_i, b_idx in enumerate(rescue_map):
                    traces[b_idx] = (
                        rescue_trace_tails[gen_i]
                        + force_suffix
                        + rescue_gens[gen_i].lstrip()
                    )

                    with _TOKENIZER_LOCK:
                        rescue_n = len(
                            self.tokenizer(
                                rescue_gens[gen_i], add_special_tokens=False
                            ).input_ids
                        )
                    token_counts[b_idx] = int(token_counts[b_idx]) + rescue_n

        with _TOKENIZER_LOCK:
            extracted_steps = [
                self.step_extractor.extract_steps(t, tokenizer=self.tokenizer)
                for t in traces
            ]

        step_spans: List = []
        for b, (trace_text, steps) in enumerate(zip(traces, extracted_steps)):
            prompt_idx = b // num_traces_per_prompt
            prompt_len_i = int(prompt_lens[prompt_idx])
            spans = self._compute_spans_from_steps(prompt_len_i, steps, trace_text)
            step_spans.append(spans)

        finish_reasons = [
            "end_think" if self._has_end_think_token(t) else "missing_end_think"
            for t in traces
        ]

        return traces, extracted_steps, step_spans, finish_reasons, token_counts

    def _get_decoder_blocks(self):
        model = self._unwrap_fsdp(self.model)

        m = getattr(model, "base_model", None)
        if m is not None:
            m = getattr(m, "model", None)
            if m is not None:
                m = getattr(m, "model", None)

        layers = getattr(m, "layers", None) if m is not None else None
        if layers is not None:
            return layers

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Could not locate decoder blocks (layers)")

    def _make_length_buckets(
        self,
        lengths: List[int],
        bucket_width: int = 256,
        token_budget: Optional[int] = None,
    ) -> List[List[int]]:

        if not lengths:
            return []

        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        buckets: List[List[int]] = []

        cur: List[int] = []
        cur_min = None
        cur_sum = 0

        for idx in order:
            L = int(lengths[idx])
            if cur_min is None:
                cur_min = L

            violates_width = (bucket_width is not None and bucket_width > 0 and (L - cur_min) > bucket_width)
            violates_budget = (token_budget is not None and token_budget > 0 and (cur_sum + L) > token_budget)

            if cur and (violates_width or violates_budget):
                buckets.append(cur)
                cur = []
                cur_min = L
                cur_sum = 0

            cur.append(idx)
            cur_sum += L

        if cur:
            buckets.append(cur)

        return buckets

    class _EarlyExitSignal(Exception):
        pass

    def _collect_selected_hidden_states(
        self,
        toks_full: dict,
        hidden_state_indices: List[int],
        step_end_indices: torch.LongTensor,
        capture_tc_input: Optional[bool] = None,
    ) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:

        needed = sorted(set(int(i) for i in hidden_state_indices))
        if not needed:
            return {}, {}

        blocks = self._get_decoder_blocks()
        captured: Dict[int, torch.Tensor] = {}
        captured_tc_input: Dict[int, torch.Tensor] = {}
        handles = []

        def _hook_factory(hs_idx: int):
            def _hook(_module, _inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out

                idx_expanded = step_end_indices.unsqueeze(-1).expand(-1, -1, hs.size(-1))
                hs_step = torch.gather(hs, dim=1, index=idx_expanded)
                captured[hs_idx] = hs_step
            return _hook

        for hs_idx in needed:
            if hs_idx <= 0:
                continue

            block_idx = hs_idx - 1
            if block_idx < 0 or block_idx >= len(blocks):
                raise IndexError(
                    f"Requested hidden_state_index={hs_idx} maps to block_idx={block_idx}, "
                    f"but decoder has {len(blocks)} blocks."
                )

            handles.append(blocks[block_idx].register_forward_hook(_hook_factory(hs_idx)))

        want_tc_input = (
            bool(self.pretrained_transcoder_repo)
            if capture_tc_input is None
            else bool(capture_tc_input)
        )
        if want_tc_input:
            def _tc_input_hook_factory(hs_idx: int):
                def _hook(_module, _inp, out):
                    tc_in = out[0] if isinstance(out, (tuple, list)) else out
                    idx_expanded = step_end_indices.unsqueeze(-1).expand(
                        -1, -1, tc_in.size(-1),
                    )
                    captured_tc_input[hs_idx] = torch.gather(
                        tc_in, dim=1, index=idx_expanded,
                    )
                return _hook

            _tc_input_layers = set(self.target_layers) & set(needed)
            for hs_idx in sorted(_tc_input_layers):
                if hs_idx <= 0:
                    continue
                block_idx = hs_idx - 1
                ln = blocks[block_idx].post_attention_layernorm
                handles.append(ln.register_forward_hook(_tc_input_hook_factory(hs_idx)))

        _needed_positive = [i for i in needed if i > 0]
        max_needed_block = max(i - 1 for i in _needed_positive) if _needed_positive else -1
        exit_block_idx = max_needed_block + 1
        _using_early_exit = exit_block_idx < len(blocks)
        if _using_early_exit:
            def _early_exit_hook(_module, _inp, _out):
                raise StudentModel._EarlyExitSignal()
            handles.append(blocks[exit_block_idx].register_forward_hook(_early_exit_hook))

        base = self._get_base_transformer()

        try:

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = base.forward(**toks_full, output_hidden_states=False, use_cache=False)
        except StudentModel._EarlyExitSignal:
            pass
        finally:
            for h in handles:
                h.remove()

        for hs_idx in needed:
            if hs_idx != 0 and hs_idx not in captured:
                raise RuntimeError(f"Failed to capture hidden state index {hs_idx} via hooks.")

        return captured, captured_tc_input

    def _unwrap_fsdp(self, m: nn.Module) -> nn.Module:
        return m.module if hasattr(m, "module") else m

    def _get_base_transformer(self):
        m = self._unwrap_fsdp(self.model)

        if hasattr(m, "peft_config"):
            m = m.base_model.model

        base = getattr(m, "model", None)
        return base if base is not None else m

    def _compute_spans_from_steps(self, prompt_len: int, steps: List[str], trace_text: str):
        if not steps or all(len(s.strip()) < 5 for s in steps):
            steps = [trace_text]

        span_list = [(0, prompt_len)]
        running_toklen = prompt_len

        for step in steps:
            with _TOKENIZER_LOCK:
                step_tok = self.tokenizer(step, return_tensors="pt",
                                        add_special_tokens=False).input_ids.shape[-1]
            end_tok = running_toklen + step_tok
            span_list.append((running_toklen, end_tok))
            running_toklen = end_tok

        return span_list[1:]

    def _get_context_limit(self) -> int:
        max_ctx = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(max_ctx, int) and 0 < max_ctx < 1_000_000:
            return max_ctx

        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
            return tok_max

        return 2048

    def _has_end_think_token(self, text: str) -> bool:
        return bool(text) and ("</think>" in text)
