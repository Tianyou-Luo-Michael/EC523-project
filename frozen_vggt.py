import torch
import clip

from vggt.models.vggt import VGGT
from vggt.models.adapter import CrossAttentionAdapter


class FrozenVGGT(VGGT):
    """
    Frozen VGGT + trainable cross-attention adapter.

    Implements the modification described for vggt/models/vggt.py:

        (a) Line 61: aggregator produces aggregated_tokens_list as normal
        (b) Right after line 61: CLIP encodes captions into 77 token hidden states
        (c) aggregated_tokens_list[-1] + CLIP tokens passed through cross-attn adapter
        (d) Adapter output replaces aggregated_tokens_list[-1] before prediction heads

    Training memory:
        VGGT aggregator (48 blocks, 1.2B params) : frozen, no_grad, no activations stored
        CLIP encoder (~400M params)               : frozen, no_grad, excluded from state_dict
        CrossAttentionAdapter (~3.4M params)      : only trainable component

    Gradient flow:
        loss → head ops → conditioned_tokens[-1] → delta → adapter.params  (correct)
                                                  ↛ VGGT aggregator         (detached)

    Dtype contract enforced under autocast(enabled=False):
        VGGT tokens (bfloat16) → .float() → float32  |
        CLIP tokens  (float16) → .float() → float32  |--> adapter --> float32
        delta                  →           float32   |
        ALL list entries       → .float() → float32 --> heads (float32)

        DPT head is a feature pyramid reading all list entries — every entry
        must be float32 before the heads see them, not just the last one.
    """

    def __init__(
        self,
        clip_model,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        adapter_dim: int = 512,
        d_text: int = 512,        # 512 for ViT-B/32, 768 for ViT-L/14
        **vggt_kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            **vggt_kwargs,
        )

        # Store CLIP bypassing nn.Module's __setattr__ so it is NOT registered
        # as a submodule. This excludes CLIP's ~400M params from:
        #   - self.parameters()  → param count shows only VGGT + adapter
        #   - self.state_dict()  → checkpoint does not contain CLIP weights
        #   - model.to(device)   → caller must move CLIP to device before passing in
        object.__setattr__(self, '_clip_model', clip_model)

        # Freeze all inherited VGGT parameters
        for p in self.parameters():
            p.requires_grad_(False)

        # Freeze CLIP parameters
        for p in self._clip_model.parameters():
            p.requires_grad_(False)

        # Adapter d_model must match what DPT/camera heads receive.
        # Heads constructed with dim_in = 2 * embed_dim = 2048.
        d_model = embed_dim * 2

        # Adapter — only trainable component.
        # Registered normally so it appears in self.parameters() and state_dict().
        self.adapter = CrossAttentionAdapter(
            d_model=d_model,
            d_text=d_text,
            adapter_dim=adapter_dim,
        )
        # Re-enable grad on adapter after the global VGGT freeze above
        for p in self.adapter.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def _encode_captions(self, captions: list[str], device) -> torch.Tensor:
        """
        Encode captions as token-level CLIP hidden states [B, 77, d_text].

        Returns CLIP's native dtype (float16 on GPU).
        Caller casts to float32 at point of use.

        Token-level rather than pooled embedding:
            Pooled → 1 KV token → softmax over 1 element always = 1
            → attention output identical for all queries → degenerates to
            a global bias, not real cross-attention.
            77 tokens → each VGGT spatial token attends differently to
            different words in the caption.
        """
        text = clip.tokenize(captions, truncate=True).to(device)
        x = self._clip_model.token_embedding(text).type(self._clip_model.dtype)
        x = x + self._clip_model.positional_embedding.type(self._clip_model.dtype)
        x = x.permute(1, 0, 2)                    # NLD → LND
        x = self._clip_model.transformer(x)
        x = x.permute(1, 0, 2)                    # LND → NLD
        x = self._clip_model.ln_final(x)           # [B, 77, d_text] float16
        return x

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        captions: list[str] = None,
    ) -> dict:
        """
        Args:
            images       : [S, 3, H, W] or [B, S, 3, H, W], range [0, 1]
            query_points : [N, 2] or [B, N, 2] — optional, for tracking
            captions     : list of B strings — optional, enables adapter.
                           If None, behaviour is identical to vanilla VGGT.

        Returns:
            Same dict as VGGT.forward() — fully backward-compatible.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # ── (a) Aggregator — vggt.py line 61 ────────────────────────────────
        # Frozen: no_grad suppresses activation storage across all 48 blocks.
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        # ── (b) + (c) CLIP encoding → cross-attention adapter ────────────────
        # Inserted immediately after line 61, as specified.
        # Skipped entirely when captions=None — vanilla VGGT behaviour preserved.
        if captions is not None:
            clip_seq = self._encode_captions(captions, images.device)

            # Hard float32 context — overrides any outer bfloat16 autocast.
            # Matches the prediction heads which also run under autocast(enabled=False).
            with torch.cuda.amp.autocast(enabled=False):

                # Cast ALL list entries to float32.
                # DPT head is a feature pyramid reading every entry in the list.
                # Mixing bfloat16 entries with one float32 entry causes silent
                # precision issues or dtype errors inside the DPT head.
                tokens_f32   = [t.float() for t in aggregated_tokens_list]
                clip_seq_f32 = clip_seq.float()                # float16 → float32

               # (c) Cross-attention adapter
                B, S, N, D = tokens_f32[-1].shape
                clip_seq_exp = clip_seq_f32.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 77, -1)

                delta = self.adapter(
                    tokens_f32[-1].reshape(B * S, N, D),
                    clip_seq_exp,
                )
                delta = delta.reshape(B, S, N, D)

                conditioned_final = tokens_f32[-1] + delta

                # (d) Replace final entry — adapter output flows to prediction heads
                aggregated_tokens_list = tokens_f32[:-1] + [conditioned_final]

        # ── (d) Prediction heads — rest of VGGT unchanged ────────────────────
        # Head params frozen (requires_grad=False) but operations are
        # differentiable — grad flows through them to delta → adapter.params.
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"]      = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"]      = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"]      = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"]   = vis
            predictions["conf"]  = conf

        if not self.training:
            predictions["images"] = images

        return predictions
