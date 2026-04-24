# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from transformers import AutoModel

from vggt.layers.block import Block


class TextEncoder(nn.Module):
    """
    Text encoder that replaces the visual Aggregator in VGGT for text-to-point-cloud generation.

    Takes tokenized text and produces aggregated_tokens_list and patch_start_idx
    in the same format expected by the downstream prediction heads (DPTHead, CameraHead, etc.).

        The encoder works in three stages:
            1. Encode text with a pretrained HF text backbone -> [B, L, d_text]
      2. Cross-attend with learnable spatial queries to produce patch tokens -> [B, num_patches, embed_dim]
      3. Prepend camera/register special tokens and refine through transformer blocks,
         collecting per-block outputs to build aggregated_tokens_list.

    Args:
        embed_dim (int): Token embedding dimension. Must match VGGT's embed_dim. Default: 1024.
        num_register_tokens (int): Number of register tokens (same as Aggregator default). Default: 4.
        patch_h (int): Number of patch rows in the virtual spatial grid. Default: 37.
        patch_w (int): Number of patch columns in the virtual spatial grid. Default: 37.
        num_heads (int): Number of attention heads in the transformer blocks. Default: 16.
        mlp_ratio (float): MLP hidden-dim expansion ratio. Default: 4.0.
        depth (int): Number of transformer blocks. Must be >= max(intermediate_layer_idx) + 1
            used in DPTHead (default indices go up to 23, so depth >= 24). Default: 24.
        text_model_name (str): Pretrained Hugging Face text backbone id.
            Default: "sentence-transformers/all-MiniLM-L6-v2".
        freeze_text_backbone (bool): If True, freeze the pretrained text
            backbone and train only downstream text-conditioning layers.
            Default: False.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_register_tokens: int = 4,
        patch_h: int = 37,
        patch_w: int = 37,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        depth: int = 24,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_text_backbone: bool = False,
    ):
        super().__init__()

        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_backbone.config.hidden_size
        if freeze_text_backbone:
            self.text_backbone.eval()
            for p in self.text_backbone.parameters():
                p.requires_grad_(False)

        # ---- Spatial layout ----
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_patches = patch_h * patch_w  # 37*37 = 1369 by default
        self.embed_dim = embed_dim

        # patch_start_idx: 1 camera token + num_register_tokens (mirrors Aggregator)
        self.patch_start_idx = 1 + num_register_tokens

        # ---- Text projection ----
        # Map pretrained text features to model embed_dim.
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # ---- Cross-attention: spatial queries <- text tokens ----
        # Each of the num_patches spatial queries learns to attend to relevant
        # parts of the text to form a 2-D feature map.
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        # ---- Learnable special tokens (camera + register) ----
        self.camera_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, embed_dim))

        # ---- Transformer blocks ----
        # Process the full token sequence (special + patch) through depth blocks.
        # Each block's output is collected to form aggregated_tokens_list, matching
        # the format produced by Aggregator.
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    qk_norm=True,
                    init_values=0.01,
                )
                for _ in range(depth)
            ]
        )

        # ---- Output projection ----
        # Aggregator concatenates frame-attention and global-attention outputs,
        # producing tokens of dim 2*embed_dim. We replicate this with a learned
        # linear projection applied at each block output.
        self.out_proj = nn.Linear(embed_dim, 2 * embed_dim)

        # Small initialisation for special tokens, matching Aggregator convention
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        nn.init.normal_(self.spatial_queries, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        """
        Encode text and produce token representations compatible with VGGT's downstream heads.

        Args:
            input_ids (torch.Tensor): Tokenized text ids with shape [B, L].
            attention_mask (torch.Tensor, optional): Binary mask with shape [B, L].
                1 for real tokens, 0 for padding. Default: None.

        Returns:
            aggregated_tokens_list (list[torch.Tensor]):
                List of ``depth`` tensors, each with shape [B, 1, P, 2*embed_dim].
                P = patch_start_idx + num_patches. The list mimics the output of
                Aggregator and can be passed directly to DPTHead, CameraHead, etc.
            patch_start_idx (int):
                Index along dim-2 where patch tokens start (= 1 + num_register_tokens).
        """
        B = input_ids.shape[0]

        # ------------------------------------------------------------------
        # Stage 1: Encode text with pretrained backbone -> [B, L, d_text]
        # ------------------------------------------------------------------
        backbone_out = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_tokens = backbone_out.last_hidden_state

        # Project text features to embed_dim -> [B, L, embed_dim]
        text_tokens = self.text_proj(text_tokens)

        # ------------------------------------------------------------------
        # Stage 2: Cross-attend spatial queries to text tokens
        #          -> [B, num_patches, embed_dim]
        # ------------------------------------------------------------------
        # key_padding_mask: True at positions the attention should IGNORE
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # [B, L]

        spatial_q = self.spatial_queries.expand(B, -1, -1)  # [B, num_patches, embed_dim]
        spatial_tokens, _ = self.cross_attn(
            spatial_q, text_tokens, text_tokens, key_padding_mask=key_padding_mask
        )  # [B, num_patches, embed_dim]
        spatial_tokens = self.cross_attn_norm(spatial_tokens)

        # ------------------------------------------------------------------
        # Stage 3: Prepend camera + register tokens -> [B, P, embed_dim]
        # ------------------------------------------------------------------
        camera_tok = self.camera_token.expand(B, -1, -1)       # [B, 1, embed_dim]
        register_tok = self.register_tokens.expand(B, -1, -1)  # [B, R, embed_dim]
        tokens = torch.cat([camera_tok, register_tok, spatial_tokens], dim=1)  # [B, P, embed_dim]

        # ------------------------------------------------------------------
        # Stage 4: Refine through transformer blocks, collect intermediate
        #          outputs to build aggregated_tokens_list
        # ------------------------------------------------------------------
        aggregated_tokens_list = []
        for block in self.blocks:
            tokens = block(tokens)
            # Project to 2*embed_dim and insert the sequence dimension S=1
            # -> [B, 1, P, 2*embed_dim], compatible with downstream heads
            out = self.out_proj(tokens).unsqueeze(1)
            aggregated_tokens_list.append(out)

        return aggregated_tokens_list, self.patch_start_idx
