import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import clip

from vggt.models.vggt import VGGT
from vggt.models.frozen_vggt import FrozenVGGT


# ── Device / dtype ────────────────────────────────────────────────────────────

device = "cuda"
dtype  = torch.bfloat16  # aggregator runs bfloat16; adapter + heads run float32


# ── Load CLIP ─────────────────────────────────────────────────────────────────

# clip_preprocess not stored — VGGT has its own image preprocessor
clip_model, _ = clip.load("ViT-B/32", device=device)


# ── Confirm VGGT token dim at runtime ─────────────────────────────────────────
# Probe the aggregator output shape before building the adapter.
# d_model = 2 * embed_dim but we verify rather than assume.

_probe = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        _dummy       = torch.zeros(1, 2, 3, 518, 518, device=device)
        _toks, _     = _probe.aggregator(_dummy)
        d_model      = _toks[-1].shape[-1]
        embed_dim    = d_model // 2       # used to construct FrozenVGGT
        print(f"VGGT token dim: {d_model}, embed_dim: {embed_dim}")  # expect 2048, 1024


# ── Build FrozenVGGT ──────────────────────────────────────────────────────────

model = FrozenVGGT(
    clip_model=clip_model,
    img_size=518,
    patch_size=14,
    embed_dim=embed_dim,   # from runtime probe, not a hardcoded assumption
    adapter_dim=512,
    d_text=512,            # ViT-B/32; use 768 for ViT-L/14
).to(device)

# Copy pretrained VGGT weights into FrozenVGGT.
# strict=False: adapter params exist in FrozenVGGT but not in the probe checkpoint.
_missing, _unexpected = model.load_state_dict(_probe.state_dict(), strict=False)
print(f"Missing keys (expected — adapter only): {_missing}")
print(f"Unexpected keys (expected — none): {_unexpected}")

# Free probe — no longer needed. Without this, two full VGGT copies sit on GPU.
del _probe
torch.cuda.empty_cache()

# Verify param counts — CLIP excluded because stored via object.__setattr__
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,}")
# Expected: ~3.4M trainable / ~1.2B total (VGGT only, CLIP excluded)

print(f"Gate at init: {model.adapter.gate.item():.4f}")   # must be 0.0000


# ── Optimizer ─────────────────────────────────────────────────────────────────

# Only adapter params — optimizer never touches frozen VGGT or CLIP weights
optimizer = AdamW(
    model.adapter.parameters(),
    lr=1e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
)

scheduler = CosineAnnealingLR(optimizer, T_max=50_000, eta_min=1e-6)


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(
    images: torch.Tensor,    # [B, S, 3, H, W]
    captions: list[str],     # B caption strings, one per scene
    gt_depth: torch.Tensor,  # [B, S, H, W]
    gt_conf: torch.Tensor,   # [B, S, H, W]  per-pixel confidence
) -> dict:
    # Frozen VGGT layers stay in eval — correct for dropout and norm layers.
    # Only the adapter is in train mode — the only thing with requires_grad=True.
    model.eval()
    model.adapter.train()
    optimizer.zero_grad()

    # ── Forward: correct captions ─────────────────────────────────────────
    # Aggregator runs bfloat16 under outer autocast.
    # Adapter + heads override to float32 via their internal autocast(enabled=False).
    with torch.cuda.amp.autocast(dtype=dtype):
        preds_correct = model(images, captions=captions)

    depth_correct = preds_correct["depth"].squeeze(-1)   # [B, S, H, W]

    # ── Forward: mismatched captions ──────────────────────────────────────
    # Shuffles captions within the batch — each scene gets a wrong description.
    # This is the comparison arm of the margin loss.
    idx_wrong      = torch.randperm(len(captions))
    captions_wrong = [captions[i] for i in idx_wrong]

    with torch.cuda.amp.autocast(dtype=dtype):
        preds_wrong = model(images, captions=captions_wrong)

    depth_wrong = preds_wrong["depth"].squeeze(-1)       # [B, S, H, W]

    # ── Losses ────────────────────────────────────────────────────────────
    valid = gt_conf > 0.5

    # Primary depth loss — gradients flow through depth_correct → adapter
    loss_depth = F.l1_loss(depth_correct[valid], gt_depth[valid])

    # Margin loss — correct caption must produce strictly lower error than wrong.
    #
    # err_wrong is DETACHED: we do not want the wrong-caption forward pass
    # to update the adapter. It exists only as a comparison target.
    #
    # err_correct_live is NOT detached: gradient flows through depth_correct
    # → conditioned_final → delta → adapter.params.
    #
    # Previous bug: both were detached → loss_caption contributed zero gradients.
    err_wrong        = (depth_wrong[valid]   - gt_depth[valid]).abs().detach()
    err_correct_live = (depth_correct[valid] - gt_depth[valid]).abs()
    margin           = 0.05
    loss_caption     = F.relu(err_correct_live - err_wrong + margin).mean()

    loss = loss_depth + 0.5 * loss_caption

    # ── Backward ─────────────────────────────────────────────────────────
    # Grad path: loss → depth_head ops → conditioned_final → delta → adapter.params
    # Frozen params accumulate no .grad — only adapter.parameters() update.
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    return {
        "loss":         loss.item(),
        "loss_depth":   loss_depth.item(),
        "loss_caption": loss_caption.item(),
        "gate":         model.adapter.gate.item(),   # watch this open up from 0
        "lr":           scheduler.get_last_lr()[0],
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(images: torch.Tensor) -> dict:
    """No captions — output identical to vanilla VGGT."""
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


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_adapter(path: str):
    """
    Save only adapter weights.
    VGGT weights are loaded from the pretrained checkpoint separately.
    CLIP weights are excluded (loaded from openai/CLIP separately).
    """
    torch.save(model.adapter.state_dict(), path)
    print(f"Saved adapter: {path}")


def load_adapter(path: str):
    model.adapter.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded adapter: {path}")
