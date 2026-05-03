import torch
import torch.nn as nn


class CrossAttentionAdapter(nn.Module):


    def __init__(
        self,
        d_model: int = 2048,
        d_text: int = 512,
        adapter_dim: int = 512,
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


        q  = self.q_proj(self.norm_q(vggt_tokens))
        kv = self.kv_proj(self.norm_kv(clip_token_seq))


        attn_out, _ = self.cross_attn(q, kv, kv)


        h = attn_out + self.ffn(self.norm_ff(attn_out))


        delta = torch.tanh(self.gate) * self.out_proj(h)

        return delta
