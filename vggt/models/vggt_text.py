# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.models.text_encoder import TextEncoder
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 text_mode=False, clip_model_name="openai/clip-vit-large-patch14", freeze_clip=True):
        super().__init__()

        self.text_mode = text_mode
        self.patch_size = patch_size

        if text_mode:
            # Virtual spatial grid: patch_h * patch_w must equal num_patches used by TextEncoder.
            # Default patch_h = patch_w = 37 gives a 518x518 equivalent (37 * 14 = 518).
            self.text_encoder = TextEncoder(
                embed_dim=embed_dim,
                patch_h=img_size // patch_size,
                patch_w=img_size // patch_size,
                clip_model_name=clip_model_name,
                freeze_clip=freeze_clip,
            )
            self.aggregator = None
        else:
            self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
            self.text_encoder = None

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor = None, query_points: torch.Tensor = None,
                input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        In image mode (text_mode=False):
            Args:
                images (torch.Tensor): Input images [S, 3, H, W] or [B, S, 3, H, W], range [0, 1].
                query_points (torch.Tensor, optional): Query points [N, 2] or [B, N, 2].

        In text mode (text_mode=True):
            Args:
                input_ids (torch.Tensor): Tokenized text ids [B, L].
                attention_mask (torch.Tensor, optional): Attention mask [B, L].

        Returns:
            dict: Predictions including world_points, world_points_conf, depth, depth_conf,
                  pose_enc (and optionally track/vis/conf if query_points supplied in image mode).
        """
        if self.text_mode:
            # ------------------------------------------------------------------
            # Text mode: encode text -> aggregated_tokens_list, patch_start_idx
            # ------------------------------------------------------------------
            aggregated_tokens_list, patch_start_idx = self.text_encoder(input_ids, attention_mask)

            B = input_ids.shape[0]
            # Create a dummy images tensor so downstream heads can read spatial shape.
            # The virtual image size matches the TextEncoder's spatial grid.
            H = self.text_encoder.patch_h * self.patch_size
            W = self.text_encoder.patch_w * self.patch_size
            images = torch.zeros(B, 1, 3, H, W, device=input_ids.device, dtype=torch.get_default_dtype())
        else:
            # ------------------------------------------------------------------
            # Image mode: encode image sequence -> aggregated_tokens_list
            # ------------------------------------------------------------------
            # If without batch dimension, add it
            if len(images.shape) == 4:
                images = images.unsqueeze(0)

            if query_points is not None and len(query_points.shape) == 2:
                query_points = query_points.unsqueeze(0)

            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

