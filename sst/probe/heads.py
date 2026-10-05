import torch
import torch.nn as nn

class TemporalProbe(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_proj: int = 512,
        n_heads: int = 8,
        n_transformer_layers: int = 4,
        max_steps: int = 48,
        dropout: float = 0.1,
        causal_mask: bool = True,
        value_head: str = "td_q",
        with_conf_head: bool = True,
        **kwargs,
    ):
        super().__init__()
        if value_head not in ("td_q", "pointer", "confidence", "value"):
            raise ValueError(
                f"[Probe] unknown value_head={value_head!r}; expected "
                f"'td_q' | 'pointer' | 'confidence' | 'value'."
            )
        if not with_conf_head and value_head != "value":
            raise ValueError(
                f"[Probe] with_conf_head=False is only valid for value_head='value' "
                f"(the standalone value probe); got value_head={value_head!r}."
            )
        self.d_input = d_input
        self.d_proj = d_proj
        self.max_steps = max_steps
        self.causal_mask = causal_mask
        self.value_head = value_head
        self.with_conf_head = bool(with_conf_head)

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_proj),
            nn.LayerNorm(d_proj),
        )

        self.pos_emb = nn.Embedding(max_steps, d_proj)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_proj,
            nhead=n_heads,
            dim_feedforward=d_proj * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )
        self.output_norm = nn.LayerNorm(d_proj)

        if value_head == "td_q":

            self.q_head = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 1),
            )
        elif value_head == "pointer":

            self.pointer_head = nn.Linear(d_proj, 1)
            self.envelope_head = nn.Linear(d_proj, 1)
            self.regime_query = nn.Linear(d_proj, 1)
            self.regime_mlp = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 3),
            )
        elif value_head == "value":

            self.value_mlp = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 1),
            )

        if self.with_conf_head:
            self.conf_head = nn.Sequential(
                nn.Linear(d_proj, d_proj),
                nn.GELU(),
                nn.Linear(d_proj, 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        length_mask: torch.Tensor | None = None,
        return_pointer: bool = False,
    ):

        B, L, _ = hidden.shape
        device = hidden.device

        if length_mask is None:
            length_mask = hidden.abs().sum(-1) > 0

        h = self.input_proj(hidden)

        positions = torch.arange(L, device=device).clamp(max=self.max_steps - 1)
        h = h + self.pos_emb(positions)

        if self.causal_mask:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                L, device=device, dtype=h.dtype,
            )
        else:
            attn_mask = None

        padding_mask = torch.zeros(B, L, device=device, dtype=h.dtype)
        padding_mask.masked_fill_(~length_mask, float("-inf"))

        h = self.transformer(
            h, mask=attn_mask, src_key_padding_mask=padding_mask,
        )
        h = self.output_norm(h)

        confidence = (torch.sigmoid(self.conf_head(h).squeeze(-1))
                      if self.with_conf_head else None)

        if self.value_head == "value":
            value = torch.sigmoid(self.value_mlp(h).squeeze(-1))
            if self.with_conf_head:
                return value, confidence
            return value

        if self.value_head == "confidence":

            return confidence

        if self.value_head == "td_q":
            value = self.q_head(h).squeeze(-1)
            return value, confidence

        if self.value_head == "pointer":
            pointer_logits = self.pointer_head(h).squeeze(-1)
            R_hat = torch.sigmoid(self.envelope_head(h).squeeze(-1))
            q = self.regime_query(h).squeeze(-1).masked_fill(~length_mask, float("-inf"))
            attn = torch.softmax(q, dim=1)
            pooled = (attn.unsqueeze(-1) * h).sum(dim=1)
            regime_logits = self.regime_mlp(pooled)
            if return_pointer:
                return pointer_logits, regime_logits, R_hat, confidence
            return pointer_logits, confidence
