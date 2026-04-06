import argparse
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as T
import clip
import numpy as np

print("SCRIPT STARTED")

from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt import FrozenVGGT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visual_util import predictions_to_glb
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ------------------------------------------------------------------------------------------------------------------ #
# This script tests the FrozenVGGT wrapper and CrossAttentionAdapter in isolation, without training.                 #
# To run: python training/test_frozen_vggt.py --image path/to/test_image.jpg --caption "a red chair near a window"   #
# ------------------------------------------------------------------------------------------------------------------ #

def load_image_as_vggt_tensor(image_path: str, img_size: int = 518) -> torch.Tensor:
    """
    Returns image tensor shaped [1, 1, 3, H, W] in [0, 1].
    - outer dim 1 = batch size B
    - second dim 1 = sequence length S (single image/frame)
    """
    image = Image.open(image_path).convert("RGB")

    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [3, H, W], float in [0, 1]
    ])

    x = tfm(image)           # [3, H, W]
    x = x.unsqueeze(0)       # [1, 3, H, W]   -> S, C, H, W
    x = x.unsqueeze(0)       # [1, 1, 3, H, W] -> B, S, C, H, W
    return x


def print_prediction_summary(preds: dict, name: str):
    print(f"\n=== {name} ===")
    print("keys:", list(preds.keys()))
    for k, v in preds.items():
        if torch.is_tensor(v):
            print(f"{k:20s} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        elif isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
            shapes = [tuple(t.shape) for t in v]
            print(f"{k:20s} list[{len(v)}] shapes={shapes}")
        else:
            print(f"{k:20s} type={type(v)}")


def preds_to_glb(preds: dict, image_tensor: torch.Tensor, filename: str):
    # Remove batch dim, move to numpy
    world_points      = preds["world_points"][0].cpu().numpy()       # [S, H, W, 3]
    world_points_conf = preds["world_points_conf"][0].cpu().numpy()  # [S, H, W]
    images            = preds["images"][0].cpu().numpy()             # [S, 3, H, W]

    # Decode pose_enc → extrinsic [S, 3, 4]
    pose_enc = preds["pose_enc"]                                     # [1, S, 9]
    extrinsic, _ = pose_encoding_to_extri_intri(pose_enc)
    extrinsic = extrinsic[0].cpu().numpy()                           # [S, 3, 4]

    scene_preds = {
        "world_points":      world_points,
        "world_points_conf": world_points_conf,
        "images":            images,
        "extrinsic":         extrinsic,
    }

    scene = predictions_to_glb(scene_preds, conf_thres=50.0, show_cam=True)
    scene.export(filename)
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a test image")
    parser.add_argument("--caption", type=str, default="a test caption")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--adapter_dim", type=int, default=512)
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    parser.add_argument("--vggt_name", type=str, default="facebook/VGGT-1B")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        amp_dtype = torch.bfloat16
    else:
        device = "cpu"
        amp_dtype = None

    print(f"Using device: {device}")

    # Load frozen CLIP text model
    clip_model, _ = clip.load(args.clip_model_name, device=device)
    clip_model.eval()

    # Load pretrained vanilla VGGT probe
    print(f"Loading VGGT checkpoint: {args.vggt_name}")
    probe = VGGT.from_pretrained(args.vggt_name).to(device)
    probe.eval()

    # Probe VGGT token dim exactly the same way your train.py does
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 3, args.img_size, args.img_size, device=device)
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                toks, _ = probe.aggregator(dummy)
        else:
            toks, _ = probe.aggregator(dummy)

        d_model = toks[-1].shape[-1]
        embed_dim = d_model // 2

    print(f"Probed d_model={d_model}, embed_dim={embed_dim}")

    # Build FrozenVGGT wrapper
    model = FrozenVGGT(
        clip_model=clip_model,
        img_size=args.img_size,
        patch_size=14,
        embed_dim=embed_dim,
        adapter_dim=args.adapter_dim,
        d_text=512 if args.clip_model_name == "ViT-B/32" else 768,
    ).to(device)

    # Copy pretrained VGGT weights into wrapper
    missing, unexpected = model.load_state_dict(probe.state_dict(), strict=False)
    print("\nload_state_dict results:")
    print("missing keys   :", missing)
    print("unexpected keys:", unexpected)

    model.eval()

    # Confirm trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {trainable:,} / {total:,}")
    print(f"Adapter gate at init: {model.adapter.gate.item():.6f}")

    # Load one image
    image_tensor = load_image_as_vggt_tensor(args.image, img_size=args.img_size).to(device)
    print(f"\nInput image tensor shape: {tuple(image_tensor.shape)}")
    print(f"Input image dtype/device: {image_tensor.dtype} / {image_tensor.device}")

    # Forward without caption
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                preds_no_caption = model(image_tensor)
        else:
            preds_no_caption = model(image_tensor)

    print_prediction_summary(preds_no_caption, "NO CAPTION")

    # Forward with caption
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                preds_with_caption = model(image_tensor, captions=[args.caption])
        else:
            preds_with_caption = model(image_tensor, captions=[args.caption])

    print_prediction_summary(preds_with_caption, "WITH CAPTION")

    # Identity check: gate=0 at init so outputs must be identical
    if "depth" in preds_no_caption and "depth" in preds_with_caption:
        diff = (preds_no_caption["depth"] - preds_with_caption["depth"]).abs().max().item()
        print(f"\nmax |depth(no_caption) - depth(with_caption)| = {diff:.8f}")

    print("\nSmoke test completed successfully.")

    # Export GLB point clouds — opens natively in Windows 3D Viewer
    if "world_points" in preds_no_caption and "world_points" in preds_with_caption:
        preds_to_glb(preds_no_caption,   image_tensor, "pointcloud_no_caption.glb")
        preds_to_glb(preds_with_caption, image_tensor, "pointcloud_with_caption.glb")


if __name__ == "__main__":
    main()