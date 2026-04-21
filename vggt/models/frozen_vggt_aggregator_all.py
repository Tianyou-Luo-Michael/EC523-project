import torch
import clip

from vggt.models.vggt import VGGT
from vggt.models.adapter import CrossAttentionAdapter


class FrozenVGGT_Aggregator_All(VGGT):
    """
    VGGT with all parameters frozen + a trainable CrossAttentionAdapter.

    Injects cross-attention adapter at all 24 levels of aggregated_tokens_list.

    CLIP is stored via object.__setattr__ to bypass nn.Module registration,
    keeping its ~400M params out of state_dict() and parameters().
    Only the adapter (~3.4M params) is trainable.

    Gradient flow: loss → heads → conditioned_tokens → delta → adapter
                                                       ↛ VGGT (frozen)

    All tokens are cast to float32 before the adapter and heads.
    """

    def __init__(
        self,
        clip_model,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        adapter_dim: int = 512,
        d_text: int = 512,
        **vggt_kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            **vggt_kwargs,
        )

        object.__setattr__(self, '_clip_model', clip_model)

        for p in self.parameters():
            p.requires_grad_(False)
        for p in self._clip_model.parameters():
            p.requires_grad_(False)

        self.adapter = CrossAttentionAdapter(
            d_model=embed_dim * 2,
            d_text=d_text,
            adapter_dim=adapter_dim,
        )
        for p in self.adapter.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def _encode_captions(self, captions: list[str], device) -> torch.Tensor:
        text = clip.tokenize(captions, truncate=True).to(device)
        x = self._clip_model.token_embedding(text).type(self._clip_model.dtype)
        x = x + self._clip_model.positional_embedding.type(self._clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self._clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        return self._clip_model.ln_final(x)

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        captions: list[str] = None,
    ) -> dict:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        if captions is not None:
            clip_seq = self._encode_captions(captions, images.device)

            with torch.cuda.amp.autocast(enabled=False):
                tokens_f32   = [t.float() for t in aggregated_tokens_list]
                clip_seq_f32 = clip_seq.float()

                target_indices = list(range(0, len(tokens_f32), 2))

                for target_idx in target_indices:
                    B, S, N, D = tokens_f32[target_idx].shape
                    clip_seq_exp = (
                        clip_seq_f32
                        .unsqueeze(1)
                        .expand(-1, S, -1, -1)
                        .reshape(B * S, 77, -1)
                    )

                    delta = self.adapter(
                        tokens_f32[target_idx].reshape(B * S, N, D),
                        clip_seq_exp,
                    ).reshape(B, S, N, D)

                    tokens_f32[target_idx] = tokens_f32[target_idx] + delta

                aggregated_tokens_list = tokens_f32

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
