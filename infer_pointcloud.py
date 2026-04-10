# Usage: python infer_pointcloud.py --images path/to/frames --adapter checkpoint.pt --out out.ply

import argparse
import struct
import os
import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms

from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt import FrozenVGGT

parser = argparse.ArgumentParser()
parser.add_argument("--images",         required=True,           help="Folder of input frames (jpg/png)")
parser.add_argument("--adapter",        required=True,           help="Path to adapter checkpoint .pt")
parser.add_argument("--caption",        default=None,            help="Optional caption for conditioning")
parser.add_argument("--out",            default="out.ply",       help="Output .ply file path")
parser.add_argument("--conf_threshold", type=float, default=0.1, help="Min confidence to keep a point")
parser.add_argument("--device",         default="cuda")
args = parser.parse_args()

device = args.device
dtype  = torch.bfloat16
EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

# Load and preprocess images
paths = sorted([
    os.path.join(args.images, f) for f in os.listdir(args.images)
    if os.path.splitext(f)[1].lower() in EXTS
])
assert paths, f"No images found in {args.images}"
print(f"Found {len(paths)} frames")

preprocess = transforms.Compose([transforms.Resize((518, 518)), transforms.ToTensor()])
images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
images = images.unsqueeze(0).to(device)  # [1, S, 3, 518, 518]

# Build model and load weights
clip_model, _ = clip.load("ViT-B/32", device=device)

print("Loading VGGT-1B...")
_probe = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        _toks, _ = _probe.aggregator(torch.zeros(1, 2, 3, 518, 518, device=device))
        embed_dim = _toks[-1].shape[-1] // 2

model = FrozenVGGT(clip_model=clip_model, img_size=518, patch_size=14,
                   embed_dim=embed_dim, adapter_dim=512, d_text=512).to(device)
model.load_state_dict(_probe.state_dict(), strict=False)
del _probe
torch.cuda.empty_cache()

# Load adapter
ck    = torch.load(args.adapter, map_location=device, weights_only=False)
state = ck["adapter"] if "adapter" in ck else ck
model.adapter.load_state_dict(state)
if "epoch" in ck:
    print(f"Adapter: epoch={ck['epoch']} steps={ck['steps']}")
model.eval()

# Run inference
print("Running inference...")
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        preds = model(images, captions=[args.caption] if args.caption else None)

# Extract and filter by confidence
pts  = preds["world_points"].squeeze(0).float().cpu().numpy()
conf = preds["world_points_conf"].squeeze(0).squeeze(-1).float().cpu().numpy()
imgs = (images.squeeze(0).permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)

S, H, W, _ = pts.shape
mask = conf > args.conf_threshold
xyz, rgb = pts[mask], imgs[mask]
print(f"Points kept: {len(xyz):,} / {S*H*W:,} (conf>{args.conf_threshold})")

# Save as binary .ply
def save_ply(path, xyz, rgb):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(len(xyz)):
            f.write(struct.pack("<fffBBB",
                float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]),
                int(rgb[i,0]),   int(rgb[i,1]),   int(rgb[i,2])))

save_ply(args.out, xyz, rgb)
print(f"Saved: {args.out}  (open with MeshLab or CloudCompare)")