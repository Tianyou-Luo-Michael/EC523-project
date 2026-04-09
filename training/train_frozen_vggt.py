import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import clip

from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt import FrozenVGGT


device = "cuda"
dtype  = torch.bfloat16  # aggregator runs bfloat16; adapter + heads run float32

# VGGT has its own image preprocessor so discard clip_preprocess
clip_model, _ = clip.load("ViT-B/32", device=device)

# Probe aggregator output shape at runtime rather than hardcoding
_probe = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        _toks, _ = _probe.aggregator(torch.zeros(1, 2, 3, 518, 518, device=device))
        d_model   = _toks[-1].shape[-1]
        embed_dim = d_model // 2
        print(f"VGGT token dim: {d_model}, embed_dim: {embed_dim}")  # expect 2048, 1024

model = FrozenVGGT(
    clip_model=clip_model,
    img_size=518,
    patch_size=14,
    embed_dim=embed_dim,
    adapter_dim=512,
    d_text=512,  # ViT-B/32; use 768 for ViT-L/14
).to(device)

# strict=False: adapter keys exist in FrozenVGGT but not in the probe checkpoint
_missing, _unexpected = model.load_state_dict(_probe.state_dict(), strict=False)
print(f"Missing (adapter only): {_missing}")
print(f"Unexpected (none): {_unexpected}")

# Free probe to avoid holding two copies of VGGT-1B on GPU
del _probe
torch.cuda.empty_cache()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,}")  # expect ~3.4M / ~1.2B
print(f"Gate at init: {model.adapter.gate.item():.4f}")  # must be 0.0000

# Optimizer over adapter params only
optimizer = AdamW(model.adapter.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.999))
scheduler = CosineAnnealingLR(optimizer, T_max=50_000, eta_min=1e-6)


def train_step(
    images: torch.Tensor,
    captions: list[str],    # one caption per scene
    gt_depth: torch.Tensor,
    gt_conf: torch.Tensor,
) -> dict:
    # Backbone stays in eval
    model.eval()
    model.adapter.train()
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(dtype=dtype):
        preds_correct = model(images, captions=captions)
    depth_correct = preds_correct["depth"].squeeze(-1)

    # Wrong-caption forward: shuffle captions so each scene gets a wrong description
    captions_wrong = [captions[i] for i in torch.randperm(len(captions))]
    with torch.cuda.amp.autocast(dtype=dtype):
        preds_wrong = model(images, captions=captions_wrong)
    depth_wrong = preds_wrong["depth"].squeeze(-1)

    valid      = gt_conf > 0.5
    loss_depth = F.l1_loss(depth_correct[valid], gt_depth[valid])

    # Margin loss: correct caption must produce lower error than wrong caption.
    err_wrong        = (depth_wrong[valid]   - gt_depth[valid]).abs().detach()
    err_correct_live = (depth_correct[valid] - gt_depth[valid]).abs()
    loss_caption     = F.relu(err_correct_live - err_wrong + 0.05).mean()

    loss = loss_depth + 0.5 * loss_caption
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    return {
        "loss":         loss.item(),
        "loss_depth":   loss_depth.item(),
        "loss_caption": loss_caption.item(),
        "gate":         model.adapter.gate.item(),
        "lr":           scheduler.get_last_lr()[0],
    }


def infer(images: torch.Tensor) -> dict:
    """Vanilla VGGT inference — no caption, adapter bypassed."""
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            return model(images)


def infer_with_caption(images: torch.Tensor, captions: list[str]) -> dict:
    """Caption-conditioned inference."""
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            return model(images, captions=captions)


def save_adapter(path: str):
    """Save adapter weights only — VGGT and CLIP are loaded separately at runtime."""
    torch.save(model.adapter.state_dict(), path)
    print(f"Saved adapter: {path}")


def load_adapter(path: str):
    ck    = torch.load(path, map_location=device, weights_only=False)
    state = ck["adapter"] if "adapter" in ck else ck
    model.adapter.load_state_dict(state)
    print(f"Loaded adapter: {path}")