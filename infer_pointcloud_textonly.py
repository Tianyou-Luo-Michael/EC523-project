"""
Infer a point cloud from a VGGTTextOnly checkpoint and a text caption.

Examples:
  python infer_pointcloud_textonly.py \
    --checkpoint logs/vggt_textonly/checkpoint_19.pt \
    --caption "a cozy kitchen with wooden cabinets" \
    --out kitchen_textonly.ply

This script supports three cases:
1) Checkpoint includes point head weights -> direct world point prediction.
2) Checkpoint has no point head but uses VGGT-initialized point head -> direct world point prediction.
3) Point head disabled explicitly -> depth backprojection fallback.
"""

import argparse
import contextlib
import os
import struct

import numpy as np
import torch
from transformers import AutoTokenizer

from vggt.models.vggt_textonly import VGGTTextOnly
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_rgb_triplet(rgb: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in rgb.split(",")]
    if len(parts) != 3:
        raise ValueError("default_color must be R,G,B")
    vals = tuple(int(v) for v in parts)
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError("default_color components must be in [0, 255]")
    return vals


def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(len(xyz)):
            f.write(
                struct.pack(
                    "<fffBBB",
                    float(xyz[i, 0]),
                    float(xyz[i, 1]),
                    float(xyz[i, 2]),
                    int(rgb[i, 0]),
                    int(rgb[i, 1]),
                    int(rgb[i, 2]),
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer point cloud from VGGTTextOnly checkpoint using a caption"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to VGGTTextOnly checkpoint (.pt)")
    parser.add_argument("--caption", required=True, help="Text prompt/caption")
    parser.add_argument("--out", default="out_textonly.ply", help="Output .ply path")
    parser.add_argument("--text_model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--vggt_model_name", default="facebook/VGGT-1B")
    parser.add_argument("--img_size", type=int, default=518, help="Virtual image size used by heads")
    parser.add_argument("--max_length", type=int, default=77, help="Tokenizer max sequence length")
    parser.add_argument("--conf_threshold", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--default_color", default="180,180,180", help="Fallback RGB color (R,G,B)")
    parser.add_argument(
        "--disable_vggt_point_head",
        action="store_true",
        help=(
            "Disable using VGGT-initialized point head when checkpoint has no point_head. "
            "If set, script falls back to depth backprojection."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    default_rgb = parse_rgb_triplet(args.default_color)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "text_encoder" not in ckpt:
        raise KeyError("Checkpoint is missing 'text_encoder' weights")

    has_ckpt_point_head = "point_head" in ckpt
    enable_camera = "camera_head" in ckpt
    enable_depth = "depth_head" in ckpt
    enable_point = has_ckpt_point_head or (not args.disable_vggt_point_head)

    if not enable_point and not (enable_depth and enable_camera):
        raise ValueError(
            "Checkpoint must contain either point_head, or both depth_head and camera_head"
        )

    print(
        "Building model with heads: "
        f"camera={enable_camera}, depth={enable_depth}, point={enable_point}"
    )
    model = VGGTTextOnly.from_pretrained_vggt(
        vggt_model_id=args.vggt_model_name,
        text_model_name=args.text_model_name,
        img_size=args.img_size,
        patch_size=14,
        enable_camera=enable_camera,
        enable_depth=enable_depth,
        enable_point=enable_point,
    ).to(device)

    model.text_encoder.load_state_dict(ckpt["text_encoder"], strict=True)
    if enable_camera and model.camera_head is not None:
        model.camera_head.load_state_dict(ckpt["camera_head"], strict=True)
    if enable_depth and model.depth_head is not None:
        model.depth_head.load_state_dict(ckpt["depth_head"], strict=True)
    if has_ckpt_point_head and model.point_head is not None:
        model.point_head.load_state_dict(ckpt["point_head"], strict=True)
        print("Loaded point_head from checkpoint")
    elif enable_point and model.point_head is not None:
        print("Checkpoint has no point_head; using original VGGT point_head weights")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
    encoded = tokenizer(
        [args.caption],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    amp_context = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    print("Running text-only forward pass...")
    with torch.no_grad():
        with amp_context:
            preds = model(input_ids=input_ids, attention_mask=attention_mask)

    if "world_points" in preds:
        pts = preds["world_points"].squeeze(0).float().cpu().numpy()  # [S, H, W, 3]
        if "world_points_conf" in preds:
            conf = preds["world_points_conf"].squeeze(0).float().cpu().numpy()  # [S, H, W]
        else:
            conf = np.ones(pts.shape[:3], dtype=np.float32)
        mode = "point_head"
    else:
        depth = preds.get("depth", None)
        pose_enc = preds.get("pose_enc", None)
        if depth is None or pose_enc is None:
            raise RuntimeError(
                "No world_points in outputs, and missing depth/pose for fallback backprojection"
            )

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=(args.img_size, args.img_size),
        )
        pts = unproject_depth_map_to_point_map(
            depth.squeeze(0).float().cpu().numpy(),
            extrinsic.squeeze(0).float().cpu().numpy(),
            intrinsic.squeeze(0).float().cpu().numpy(),
        )
        if "depth_conf" in preds:
            conf = preds["depth_conf"].squeeze(0).float().cpu().numpy()  # [S, H, W]
        else:
            conf = np.ones(pts.shape[:3], dtype=np.float32)
        mode = "depth_backprojection"

    S, H, W, _ = pts.shape
    mask = conf > args.conf_threshold
    xyz = pts[mask]

    if xyz.shape[0] == 0:
        raise RuntimeError(
            f"No points survived confidence threshold {args.conf_threshold}. "
            "Try lowering --conf_threshold."
        )

    rgb = np.tile(np.asarray(default_rgb, dtype=np.uint8), (xyz.shape[0], 1))

    print(f"Inference mode: {mode}")
    print(f"Points kept: {len(xyz):,} / {S * H * W:,} (conf>{args.conf_threshold})")

    save_ply(args.out, xyz, rgb)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
