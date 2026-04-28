# Text-Conditioned 3D Reconstruction with VGGT

**EC523 Deep Learning — Boston University**  
Timucin Erbas, Leroy Adisaputro, Azsadur Rakin, Tianyou Luo

An extension of [VGGT (CVPR 2025)](https://arxiv.org/abs/2503.11651) that injects natural language captions into the reconstruction pipeline via a lightweight cross-attention adapter. The frozen VGGT backbone (~1.2B params) is unchanged; only the adapter (~3.4M params) is trained.

## Overview

VGGT reconstructs 3D scenes from images, estimating camera poses, depth maps, and point clouds in a single forward pass. We augment this pipeline by conditioning the internal token representations on a scene caption, allowing semantic text information to guide 3D reconstruction without retraining the full model.

A `CrossAttentionAdapter` is inserted between VGGT's aggregator and its prediction heads. It fuses the final aggregated visual tokens with token-level CLIP text embeddings, producing a residual update that conditions the representation on the caption. A scalar gate initialized to zero ensures the adapter starts as an identity mapping and opens gradually during training.

```
Images  -> VGGT Aggregator (frozen) -> visual tokens -+
Caption -> CLIP Encoder (frozen)    -> 77 text tokens  +-> CrossAttentionAdapter -> conditioned tokens -> heads
```

## Project Structure

```
EC523-project/
├── infer_pointcloud.py          # Inference script, outputs .ply point cloud
├── vggt/
│   ├── models/
│   │   ├── vggt.py              # Original VGGT (unchanged)
│   │   ├── aggregator.py        # Original aggregator (unchanged)
│   │   ├── frozen_vggt.py       # FrozenVGGT wrapper (ours)
│   │   └── adapter.py           # CrossAttentionAdapter (ours)
│   ├── heads/                   # Original prediction heads (unchanged)
│   ├── layers/                  # Original transformer layers (unchanged)
│   └── utils/                   # Original utilities (unchanged)
└── training/
    ├── train_frozen_vggt.py     # Training entry point
    ├── test_frozen_vggt.py      # Evaluation script
    ├── trainer.py               # Base trainer (unchanged)
    ├── loss.py                  # Multitask loss (unchanged)
    └── data/
        └── datasets/co3d.py     # CO3D dataset loader
```

## Setup

```bash
git clone https://github.com/Tianyou-Luo-Michael/EC523-project.git
cd EC523-project
pip install -r requirements.txt
```

Requires a CUDA GPU with at least 16GB VRAM.

## Inference

Without caption (adapter bypassed, identical to vanilla VGGT):
```bash
python infer_pointcloud.py \
  --images  examples/llff_fern/images \
  --adapter checkpoint_9.pt \
  --out     scene_no_caption.ply
```

With caption (adapter active):
```bash
python infer_pointcloud.py \
  --images   examples/llff_fern/images \
  --adapter  checkpoint_9.pt \
  --caption  "a potted fern plant that is green" \
  --out      scene_with_caption.ply
```

| Argument | Default | Description |
|---|---|---|
| `--images` | required | Folder of input frames (jpg/png) |
| `--adapter` | required | Path to adapter checkpoint .pt |
| `--caption` | None | Scene description, omit for vanilla VGGT |
| `--out` | `out.ply` | Output file path |
| `--conf_threshold` | 0.1 | Min confidence to keep a point |
| `--device` | `cuda` | Device |

Output `.ply` files can be opened with MeshLab or CloudCompare.

## Training

Training was run on the Boston University SCC cluster using a curated captioned subset of Co3Dv2, with 619 training sequences and 68 held-out test sequences across 35 categories. The dataset includes RGB images, depth maps, and sequence-level captions generated from multiple views.

The adapter is trained with two losses:
- **Depth L1 loss**: primary supervision on correct-caption predictions
- **Margin loss**: penalizes correct-caption depth predictions that are worse than wrong-caption predictions, encouraging the adapter to use the caption meaningfully

Only adapter parameters are updated. VGGT and CLIP remain fully frozen. Checkpoints save adapter weights only (~54MB).

```bash
python training/train_frozen_vggt.py
```

## Adapter Architecture

The `CrossAttentionAdapter` (~3.4M params):

1. Layer norm + linear projection maps VGGT visual tokens (dim 2048) and CLIP text tokens (dim 512) into a shared bottleneck (dim 512)
2. 8-head cross-attention with VGGT tokens as queries and 77 CLIP tokens as keys/values
3. FFN with residual in bottleneck space
4. Linear projection back to dim 2048
5. Scalar gate (`tanh(gate)`, init=0) scales the output, guaranteeing identity at initialization

## Citation

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={CVPR},
  year={2025}
}
```