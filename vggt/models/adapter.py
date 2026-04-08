import torch
import torch.nn as nn


class CrossAttentionAdapter(nn.Module):
    """
    Lightweight cross-attention adapter (~3.4M params).

    Injects CLIP token-level text embeddings into VGGT's aggregated token
    space via a bottlenecked cross-attention module.

    Architecture:
        Pre-norm on inputs (standard: norm before projection)
        Bottleneck: d_model → adapter_dim, d_text → adapter_dim
        Cross-attention in adapter_dim: VGGT tokens (Q) x CLIP tokens (K, V)
        FFN with residual in adapter_dim
        Output projection: adapter_dim → d_model
        Scalar gate (init=0) → delta=0 at init, identity guaranteed

    Param count (d_model=2048, d_text=512, adapter_dim=512, num_heads=8):
        norm_q:     2 x 2048         ~ 0.00M
        norm_kv:    2 x 512          ~ 0.00M
        q_proj:     2048 x 512       = 1.05M
        kv_proj:    512  x 512       = 0.26M
        cross_attn: 4 x 512 x 512    = 1.05M
        norm_ff:    2 x 512          ~ 0.00M
        ffn:        512→1024→512     = 1.05M
        out_proj:   512 x 2048       = 1.05M
        gate:       1                ~ 0.00M
        Total:                       ~ 3.4M

    Identity at init — why gate=0 alone is sufficient, NOT gate=0 + zero out_proj:

        delta = tanh(gate) * out_proj(h)

        At init: delta = tanh(0) * out_proj(h) = 0 * nonzero = 0  (identity)

        d_loss/d_gate = d_loss/d_delta * (1 - tanh^2(0)) * out_proj(h)
                      = d_loss/d_delta * 1 * nonzero != 0           (can learn)

        If out_proj were also zero-inited:
            out_proj(h) = 0 → all gradients at init = 0 → complete gradient death.
        That is why out_proj is NOT zero-inited here.
    """

    def __init__(
        self,
        d_model: int = 2048,      # VGGT token dim — confirmed at runtime in train.py
        d_text: int = 512,        # CLIP text dim (ViT-B/32=512, ViT-L/14=768)
        adapter_dim: int = 512,   # bottleneck dim — controls param count
        num_heads: int = 8,
    ):
        super().__init__()

        # Pre-norm on inputs before projection (standard pre-norm order)
        self.norm_q  = nn.LayerNorm(d_model)    # normalises VGGT tokens
        self.norm_kv = nn.LayerNorm(d_text)     # normalises CLIP tokens

        # Bottleneck projections into adapter_dim
        self.q_proj  = nn.Linear(d_model, adapter_dim)
        self.kv_proj = nn.Linear(d_text,  adapter_dim)

        # Cross-attention in bottleneck dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        # Pre-norm before FFN (in bottleneck dim)
        self.norm_ff = nn.LayerNorm(adapter_dim)

        # FFN in bottleneck dim
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )

        # Project back to d_model — NOT zero-inited (see docstring above)
        self.out_proj = nn.Linear(adapter_dim, d_model)

        # Scalar gate: tanh(0)=0 → delta=0 at init (identity)
        # out_proj produces nonzero output so gate gradient is nonzero at init
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vggt_tokens: torch.Tensor,      # [B, N, d_model]  float32
        clip_token_seq: torch.Tensor,   # [B, 77, d_text]  float32
    ) -> torch.Tensor:
        """
        Returns delta [B, N, d_model] to be added to VGGT token stream.

        At init: gate=0 → delta=0 → identity.
        During training: gate opens gradually, adapter activates.
        """
        # Pre-norm then project to bottleneck (standard pre-norm order)
        q  = self.q_proj(self.norm_q(vggt_tokens))       # [B, N, adapter_dim]
        kv = self.kv_proj(self.norm_kv(clip_token_seq))  # [B, 77, adapter_dim]

        # Cross-attention: each VGGT spatial token attends to all 77 text tokens
        attn_out, _ = self.cross_attn(q, kv, kv)         # [B, N, adapter_dim]

        # FFN with residual in bottleneck dim
        h = attn_out + self.ffn(self.norm_ff(attn_out))  # [B, N, adapter_dim]

        # Gate the FULL output — single scalar ensures complete identity at init
        delta = torch.tanh(self.gate) * self.out_proj(h)  # [B, N, d_model]

        return delta
