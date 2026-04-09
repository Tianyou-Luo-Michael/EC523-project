import argparse
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as T
import clip
import numpy as np

from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt import FrozenVGGT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from visual_util import predictions_to_glb
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_image(image_path: str, img_size: int = 518) -> torch.Tensor:
    """Returns [1, 1, 3, H, W] float tensor in [0, 1]."""
    image = Image.open(image_path).convert("RGB")
    x = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])(image)
    return x.unsqueeze(0).unsqueeze(0)


def print_summary(preds: dict, name: str):
    print(f"\n=== {name} ===")
    for k, v in preds.items():
        if torch.is_tensor(v):
            print(f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
            print(f"  {k:20s} list[{len(v)}] shapes={[tuple(t.shape) for t in v]}")


def export_glb(preds: dict, filename: str):
    """Export world_points prediction as a .glb point cloud."""
    extrinsic, _ = pose_encoding_to_extri_intri(preds["pose_enc"])
    scene_preds = {
        "world_points":      preds["world_points"][0].cpu().numpy(),
        "world_points_conf": preds["world_points_conf"][0].cpu().numpy(),
        "images":            preds["images"][0].cpu().numpy(),
        "extrinsic":         extrinsic[0].cpu().numpy(),
    }
    predictions_to_glb(scene_preds, conf_thres=50.0, show_cam=True).export(filename)
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",           type=str, required=True)
    parser.add_argument("--caption",         type=str, default="a test caption")
    parser.add_argument("--img_size",        type=int, default=518)
    parser.add_argument("--adapter_dim",     type=int, default=512)
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32")
    parser.add_argument("--vggt_name",       type=str, default="facebook/VGGT-1B")
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" else None
    print(f"Device: {device}")

    clip_model, _ = clip.load(args.clip_model_name, device=device)
    clip_model.eval()

    probe = VGGT.from_pretrained(args.vggt_name).to(device)
    probe.eval()

    # Probe aggregator to get token dim at runtime
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 3, args.img_size, args.img_size, device=device)
        ctx   = torch.cuda.amp.autocast(dtype=amp_dtype) if device == "cuda" else torch.no_grad()
        with ctx:
            toks, _ = probe.aggregator(dummy)
        d_model   = toks[-1].shape[-1]
        embed_dim = d_model // 2
    print(f"d_model={d_model}, embed_dim={embed_dim}")

    model = FrozenVGGT(
        clip_model=clip_model,
        img_size=args.img_size,
        patch_size=14,
        embed_dim=embed_dim,
        adapter_dim=args.adapter_dim,
        d_text=512 if args.clip_model_name == "ViT-B/32" else 768,
    ).to(device)

    missing, unexpected = model.load_state_dict(probe.state_dict(), strict=False)
    print(f"Missing (adapter only): {missing}")
    print(f"Unexpected (none): {unexpected}")
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")
    print(f"Gate at init: {model.adapter.gate.item():.6f}")  # must be 0.000000

    image_tensor = load_image(args.image, args.img_size).to(device)

    def run(captions=None):
        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    return model(image_tensor, captions=captions)
            return model(image_tensor, captions=captions)

    preds_no_caption   = run()
    preds_with_caption = run(captions=[args.caption])

    print_summary(preds_no_caption,   "NO CAPTION")
    print_summary(preds_with_caption, "WITH CAPTION")

    # At init gate=0, so both outputs must be identical
    if "depth" in preds_no_caption:
        diff = (preds_no_caption["depth"] - preds_with_caption["depth"]).abs().max().item()
        print(f"\nmax depth diff (expect 0 at init): {diff:.8f}")

    if "world_points" in preds_no_caption:
        export_glb(preds_no_caption,   "pointcloud_no_caption.glb")
        export_glb(preds_with_caption, "pointcloud_with_caption.glb")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()