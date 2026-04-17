# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGTTextOnly
============
A text-only variant of VGGT designed to be trained via representation
alignment with a frozen image encoder.

Training objective
------------------
Given a batch of (image_sequence, text_caption) pairs:
  1. Alignment loss  — minimise the distance between text tokens
     (from TextEncoder) and image tokens (from a frozen Aggregator).
     Both an MSE term and a cosine-similarity term are used.
  2. Task loss       — optional 3D prediction losses (depth, world
     points, camera pose) computed from the text-derived tokens.

Inference
---------
Only text_encoder + downstream prediction heads are used.
No images are required at inference time.

Typical usage
-------------
>>> model = VGGTTextOnly.from_pretrained_vggt("facebook/VGGT-1B")
>>> model.freeze_aggregator()          # freeze image teacher
>>> # ... training loop ...
>>> model.text_forward(input_ids, attention_mask)   # inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.models.text_encoder import TextEncoder
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGTTextOnly(nn.Module, PyTorchModelHubMixin):
    """
    Text-only VGGT model.

    At inference time the model receives tokenised text and produces the
    same 3-D predictions (depth, world points, camera pose) as the image
    VGGT, relying solely on text representations.

    At training time the model additionally receives image frames.  A
    frozen copy of the VGGT image Aggregator produces per-block token
    tensors that serve as the *teacher* signal.  The trainable
    TextEncoder (student) is optimised to reproduce those tokens.

    Args:
        img_size (int): Spatial resolution passed to Aggregator / used to
            derive the virtual patch grid for TextEncoder.  Default: 518.
        patch_size (int): Patch size.  Default: 14.
        embed_dim (int): Token embedding dimension.  Default: 1024.
        enable_camera (bool): Include CameraHead.  Default: True.
        enable_point (bool): Include DPTHead for 3-D world points.
            Default: True.
        enable_depth (bool): Include DPTHead for depth.  Default: True.
        clip_model_name (str): HuggingFace model name for the CLIP text
            backbone inside TextEncoder.
            Default: ``"openai/clip-vit-large-patch14"``.
        freeze_clip (bool): Freeze CLIP weights inside TextEncoder.
            Default: True.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_depth: bool = True,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        freeze_clip: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_h = img_size // patch_size  # 37 for 518 / 14
        patch_w = img_size // patch_size

        # ------------------------------------------------------------------
        # Student: trainable text encoder
        # ------------------------------------------------------------------
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            patch_h=patch_h,
            patch_w=patch_w,
            clip_model_name=clip_model_name,
            freeze_clip=freeze_clip,
        )

        # ------------------------------------------------------------------
        # Teacher: image aggregator used only during training.
        # Call freeze_aggregator() before the training loop.
        # ------------------------------------------------------------------
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # ------------------------------------------------------------------
        # Downstream prediction heads (operate on text tokens at inference,
        # and on text tokens during training for task supervision).
        # ------------------------------------------------------------------
        self.camera_head = (
            CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        )
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
            if enable_depth
            else None
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def freeze_aggregator(self) -> None:
        """Freeze all Aggregator parameters (call once before training)."""
        self.aggregator.eval()
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

    def freeze_heads(self) -> None:
        """Optionally freeze downstream heads to train only the text encoder."""
        for head in (self.camera_head, self.depth_head, self.point_head):
            if head is not None:
                for p in head.parameters():
                    p.requires_grad_(False)

    def unfreeze_heads(self) -> None:
        """Unfreeze downstream heads for end-to-end fine-tuning."""
        for head in (self.camera_head, self.depth_head, self.point_head):
            if head is not None:
                for p in head.parameters():
                    p.requires_grad_(True)

    @classmethod
    def from_pretrained_vggt(
        cls,
        vggt_model_id: str = "facebook/VGGT-1B",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        freeze_clip: bool = True,
        **init_kwargs,
    ) -> "VGGTTextOnly":
        """
        Construct a VGGTTextOnly initialised from a pretrained VGGT checkpoint.

        The Aggregator and downstream head weights are copied from the
        pretrained VGGT model.  The TextEncoder is initialised from
        scratch (its CLIP backbone is loaded from HuggingFace).

        Args:
            vggt_model_id (str): HuggingFace model hub id for the base
                VGGT checkpoint (e.g. ``"facebook/VGGT-1B"``).
            clip_model_name (str): CLIP backbone for the TextEncoder.
            freeze_clip (bool): Freeze CLIP inside TextEncoder.
            **init_kwargs: Forwarded to ``VGGTTextOnly.__init__``.

        Returns:
            VGGTTextOnly: Model with pretrained aggregator + head weights.
        """
        from vggt.models.vggt import VGGT

        # Load the base model to extract its weights
        base = VGGT.from_pretrained(vggt_model_id)

        # Infer dimensions from the base model
        embed_dim = base.aggregator.embed_dim if hasattr(base.aggregator, "embed_dim") else 1024
        img_size = init_kwargs.pop("img_size", 518)
        patch_size = init_kwargs.pop("patch_size", 14)

        model = cls(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            clip_model_name=clip_model_name,
            freeze_clip=freeze_clip,
            **init_kwargs,
        )

        # Copy aggregator weights
        model.aggregator.load_state_dict(base.aggregator.state_dict())

        # Copy head weights where available
        for attr in ("camera_head", "depth_head", "point_head"):
            src = getattr(base, attr, None)
            dst = getattr(model, attr, None)
            if src is not None and dst is not None:
                dst.load_state_dict(src.state_dict())

        del base
        torch.cuda.empty_cache()
        return model

    # ------------------------------------------------------------------
    # Internal: build dummy image tensor for downstream heads
    # ------------------------------------------------------------------

    def _dummy_images(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create a zero-filled image tensor of the correct spatial shape.

        The downstream DPT heads use the ``images`` argument only to read
        its spatial dimensions for feature-map interpolation.  A zero
        tensor is sufficient.

        Shape: [B, 1, 3, H, W]
        """
        H = self.text_encoder.patch_h * self.patch_size
        W = self.text_encoder.patch_w * self.patch_size
        return torch.zeros(
            batch_size, 1, 3, H, W,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        images: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids (torch.Tensor): Tokenised text, shape [B, L].
            attention_mask (torch.Tensor, optional): Binary mask [B, L].
                1 for real tokens, 0 for padding.  Default: None.
            images (torch.Tensor, optional): Input frames for alignment
                supervision during training.  Shape [B, S, 3, H, W] or
                [S, 3, H, W].  Only used when ``self.training`` is True.
                Ignored at inference.  Default: None.

        Returns:
            dict: A prediction dictionary with the following keys:

            Always present:
              - ``text_tokens_list`` (list[Tensor[B, 1, P, 2D]]): Per-block
                text token tensors from the TextEncoder.
              - ``pose_enc`` / ``pose_enc_list``: Camera pose encoding
                (if camera_head is enabled).
              - ``depth`` / ``depth_conf``: Depth prediction (if depth_head
                is enabled).
              - ``world_points`` / ``world_points_conf``: 3-D world point
                prediction (if point_head is enabled).

            Present only during training when images are provided:
              - ``image_tokens_list`` (list[Tensor[B, 1, P, 2D]]): Per-block
                *detached* image token tensors from the frozen Aggregator.
                Use these together with ``text_tokens_list`` to compute the
                alignment loss (see ``alignment_loss()`` below).

            Present only during inference (``self.training == False``):
              - ``images``: The zero dummy image tensor (for visualisation
                compatibility).
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # ------------------------------------------------------------------
        # 0. Normalise the images tensor and derive spatial / sequence info.
        #    This is done now (before the DPT head calls) so that:
        #      (a) The DPT / camera heads use the real image H×W as their
        #          output-resolution reference, avoiding the 518×518 dummy
        #          mismatch when input images are non-square (e.g. 476×518).
        #      (b) Predictions are expanded along S so their shape matches
        #          the GT tensors (which have S = number of input frames).
        # ------------------------------------------------------------------
        if images is not None:
            if images.dim() == 4:
                images = images.unsqueeze(0)  # [S, C, H, W] → [1, S, C, H, W]
            S = images.shape[1]
            # First frame only: gives DPT head the correct H, W reference.
            ref_images = images[:, :1].contiguous()  # [B, 1, 3, H, W]
        else:
            S = 1
            ref_images = self._dummy_images(B, device)  # [B, 1, 3, 518, 518]

        # ------------------------------------------------------------------
        # 1. Text encoding (student) — always executed
        # ------------------------------------------------------------------
        text_tokens_list, patch_start_idx = self.text_encoder(
            input_ids, attention_mask
        )

        # ------------------------------------------------------------------
        # 2. Downstream predictions from text tokens
        #    Heads produce outputs with the virtual S=1 sequence dimension.
        #    We immediately expand to the real S so downstream losses can
        #    index predictions and GT with the same mask shape.
        # ------------------------------------------------------------------
        predictions: dict = {}

        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(text_tokens_list)
                # pose_enc: [B, 1, 9] → [B, S, 9]
                predictions["pose_enc"] = pose_enc_list[-1].expand(-1, S, -1)
                predictions["pose_enc_list"] = [
                    t.expand(-1, S, -1) for t in pose_enc_list
                ]

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    text_tokens_list,
                    images=ref_images,
                    patch_start_idx=patch_start_idx,
                )
                # depth: [B, 1, H, W, 1] → [B, S, H, W, 1]
                predictions["depth"] = depth.expand(-1, S, -1, -1, -1)
                # depth_conf: [B, 1, H, W] → [B, S, H, W]
                predictions["depth_conf"] = depth_conf.expand(-1, S, -1, -1)

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    text_tokens_list,
                    images=ref_images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d.expand(-1, S, -1, -1, -1)
                predictions["world_points_conf"] = pts3d_conf.expand(-1, S, -1, -1)

        predictions["text_tokens_list"] = text_tokens_list

        # ------------------------------------------------------------------
        # 3. Image encoding (teacher) — only during training with images
        # ------------------------------------------------------------------
        if self.training and images is not None:
            # Run frozen aggregator without gradient tracking
            with torch.no_grad():
                image_tokens_list, _ = self.aggregator(images)
                # image_tokens_list: list[Tensor[B, S, P, 2D]]
                # Average over the frame dimension S → [B, 1, P, 2D] so that
                # it matches the TextEncoder's single-sequence output.
                image_tokens_list = [
                    t.mean(dim=1, keepdim=True).detach()
                    for t in image_tokens_list
                ]

            predictions["image_tokens_list"] = image_tokens_list

        if not self.training:
            predictions["images"] = ref_images

        return predictions


# ---------------------------------------------------------------------------
# Standalone alignment loss helper
# ---------------------------------------------------------------------------

def alignment_loss(
    text_tokens_list: list,
    image_tokens_list: list,
    mse_weight: float = 0.5,
    cosine_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute the alignment loss between text and image token sequences.

    This is the primary training signal for VGGTTextOnly.  It encourages
    the TextEncoder to produce per-block token representations that
    match the frozen Aggregator's representations of the corresponding
    image frames.

    The loss is an average over all transformer blocks of a weighted sum
    of MSE loss and cosine-similarity loss.

    Args:
        text_tokens_list (list[Tensor]): Per-block text tokens from
            VGGTTextOnly's forward.  Each element has shape
            [B, 1, P, 2*embed_dim].
        image_tokens_list (list[Tensor]): Per-block image tokens from
            VGGTTextOnly's forward (``image_tokens_list`` key in the
            returned dict).  Same shape as text_tokens_list.
        mse_weight (float): Weight of the MSE term.  Default: 0.5.
        cosine_weight (float): Weight of the cosine-similarity term.
            Default: 0.5.

    Returns:
        torch.Tensor: Scalar alignment loss.

    Example::

        preds = model(input_ids, attention_mask, images=frames)
        loss_align = alignment_loss(
            preds["text_tokens_list"],
            preds["image_tokens_list"],
        )
        loss_task = task_loss_fn(preds, batch)
        loss = loss_align + loss_task
        loss.backward()
    """
    assert len(text_tokens_list) == len(image_tokens_list), (
        "text and image token lists must have the same number of blocks"
    )

    mse_total = torch.tensor(0.0, device=text_tokens_list[0].device)
    cos_total = torch.tensor(0.0, device=text_tokens_list[0].device)

    for text_t, img_t in zip(text_tokens_list, image_tokens_list):
        # text_t: [B, S_t, P_t, D]   img_t: [B, S_i, P_i, D]
        # P_t may differ from P_i when input images are not 518x518.
        # Mean-pool over all token positions -> [B, D] so the loss is
        # always numerically valid regardless of resolution.
        B, _, _, D = text_t.shape
        t_pooled = text_t.reshape(B, -1, D).mean(dim=1)   # [B, D]
        i_pooled = img_t.reshape(B, -1, D).mean(dim=1)    # [B, D]

        if mse_weight > 0:
            mse_total = mse_total + F.mse_loss(t_pooled, i_pooled)

        if cosine_weight > 0:
            # cosine_similarity returns values in [-1, 1]; loss in [0, 2]
            cos_total = cos_total + (
                1.0 - F.cosine_similarity(t_pooled, i_pooled, dim=-1)
            ).mean()

    n = len(text_tokens_list)
    return mse_weight * mse_total / n + cosine_weight * cos_total / n
