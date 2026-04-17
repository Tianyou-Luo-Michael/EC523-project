import torch
import torch.nn as nn


class CrossAttentionAdapter(nn.Module):
    """
    Lightweight cross-attention adapter (~3.4M params) that injects CLIP
    token-level text embeddings into VGGT's aggregated token space.

    At init: gate=0 → tanh(0)=0 → delta=0 (identity, no effect on VGGT).
    During training: gate opens gradually, adapter activates.

    out_proj is NOT zero-inited — zeroing it would kill all gradients at init
    since grad flows through gate * out_proj(h), and 0 * 0 = 0 everywhere.
    """

    def __init__(
        self,
        d_model: int = 2048,    # VGGT token dim
        d_text: int = 512,      # CLIP text dim (ViT-B/32=512, ViT-L/14=768)
        adapter_dim: int = 512, # bottleneck dim
        num_heads: int = 8,
    ):
        super().__init__()

        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_text)

        self.q_proj  = nn.Linear(d_model, adapter_dim)
        self.kv_proj = nn.Linear(d_text,  adapter_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.norm_ff = nn.LayerNorm(adapter_dim)
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 2),
            nn.GELU(),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )

        self.out_proj = nn.Linear(adapter_dim, d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vggt_tokens: torch.Tensor,
        clip_token_seq: torch.Tensor,
    ) -> torch.Tensor:
        # Project both inputs into bottleneck dim
        q  = self.q_proj(self.norm_q(vggt_tokens))
        kv = self.kv_proj(self.norm_kv(clip_token_seq))

        # Each VGGT spatial token attends to all 77 CLIP text tokens
        attn_out, _ = self.cross_attn(q, kv, kv)

        # FFN with residual
        h = attn_out + self.ffn(self.norm_ff(attn_out))

        # Scalar gate controls adapter strength
        return torch.tanh(self.gate) * self.out_proj(h)