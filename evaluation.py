import gzip, json, math, os, sys
import numpy as np
import torch
import clip

from PIL import Image
from torchvision import transforms
from vggt.models.vggt import VGGT


CO3D_DIR     = # "/path/to/co3d_curated"
ANNO_DIR     = # "/path/to/co3d_curated_anno_split"
CAPTION_FILE = # "/path/to/co3d_curated_captions_test.jsonl"
OUT_FILE     = # "logs/eval_results.json"


NUM_FRAMES = 5
IMG_SIZE   = 518
CHAMFER_N  = 2000
device     = "cuda"
dtype      = torch.bfloat16


EXPERIMENTS = {
    "vanilla": (
        None, None, None, 512,
    ),
    "agg@24": (
        "vggt.models.frozen_vggt",
        "FrozenVGGT",
        # "/path/to/agg24/checkpoint.pt",
        512,
    ),
    "agg@0": (
        "vggt.models.frozen_vggt_aggregator_0",
        "FrozenVGGT_Aggregator_0",
        # "/path/to/agg0/checkpoint.pt",
        512,
    ),
    "agg@12": (
        "vggt.models.frozen_vggt_aggregator_12",
        "FrozenVGGT_Aggregator_12",
        # "/path/to/agg12/checkpoint.pt",
        512,
    ),
    "multi-3": (
        "vggt.models.frozen_vggt_aggregator_3",
        "FrozenVGGT_Aggregator_3",
        # "/path/to/multi3/checkpoint.pt",
        512,
    ),
    "every-2": (
        "vggt.models.frozen_vggt_aggregator_everyother",
        "FrozenVGGT_Aggregator_Everyother",
        # "/path/to/every2/checkpoint.pt",
        512,
    ),
    "agg@dpt": (
        "vggt.models.frozen_vggt_aggregator_dpt",
        "FrozenVGGT_Aggregator_DPT",
        # "/path/to/aggdpt/checkpoint.pt",
        512,
    ),
    "multi-dpt": (
        "vggt.models.frozen_vggt_multi_alldpt",
        "FrozenVGGT_Multi_AllDPT",
        # "/path/to/multidpt/checkpoint.pt",
        512,
    ),
    "dim128": (
        "vggt.models.frozen_vggt_aggregator_dpt",
        "FrozenVGGT_Aggregator_DPT",
        # "/path/to/dim128/checkpoint.pt",
        128,
    ),
    "dim256": (
        "vggt.models.frozen_vggt_aggregator_dpt",
        "FrozenVGGT_Aggregator_DPT",
        # "/path/to/dim256/checkpoint.pt",
        256,
    ),
    "dim1024": (
        "vggt.models.frozen_vggt_aggregator_dpt",
        "FrozenVGGT_Aggregator_DPT",
        # "/path/to/dim1024/checkpoint.pt",
        1024,
    ),
}


preprocess       = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
depth_preprocess = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_sequences(anno_dir, co3d_dir, split="test"):
    sequences = []
    for fname in sorted(os.listdir(anno_dir)):
        if not fname.endswith(f"_{split}.jgz"):
            continue
        with gzip.open(os.path.join(anno_dir, fname), "r") as f:
            annotation = json.loads(f.read())
        category = fname.replace(f"_{split}.jgz", "")
        for seq_name, frames in annotation.items():
            seq_dir   = os.path.join(co3d_dir, category, seq_name)
            img_dir   = os.path.join(seq_dir, "images")
            depth_dir = os.path.join(seq_dir, "depths")
            mask_dir  = os.path.join(seq_dir, "depth_masks")
            if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
                continue
            img_paths = sorted([
                os.path.join(img_dir, f) for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower() in EXTS
            ])
            depth_paths = sorted([
                os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            ])
            mask_paths = sorted([
                os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            ]) if os.path.isdir(mask_dir) else []
            if len(img_paths) < 3 or len(depth_paths) < 3:
                continue
            sequences.append({
                "seq_name": seq_name,
                "category": category,
                "images":   img_paths,
                "depths":   depth_paths,
                "masks":    mask_paths,
            })
    return sequences

def load_images(paths, indices):
    frames = torch.stack([preprocess(Image.open(paths[i]).convert("RGB")) for i in indices])
    return frames.unsqueeze(0)

def load_depths(depth_paths, mask_paths, indices):
    depths, masks = [], []
    for i in indices:
        d = depth_preprocess(Image.open(depth_paths[i]).convert("L"))
        depths.append(d)
        if mask_paths and i < len(mask_paths):
            m = depth_preprocess(Image.open(mask_paths[i]).convert("L"))
            masks.append(m > 0.5)
        else:
            masks.append(torch.ones_like(d, dtype=torch.bool))
    return torch.stack(depths), torch.stack(masks)


def compute_l1(pred_depth, gt_depth, mask):
    pred   = pred_depth.squeeze(0).squeeze(-1).float().cpu()
    gt     = gt_depth.squeeze(1).float()
    m      = mask.squeeze(1)
    gt_max = gt[m].max()
    if gt_max <= 0 or m.sum() == 0:
        return float("nan")
    gt_norm = gt / gt_max
    return (pred[m] - gt_norm[m]).abs().mean().item()

def compute_chamfer(pred_world_points, pred_conf, gt_depth, gt_mask, n=CHAMFER_N):
    """
    Bidirectional Chamfer distance between predicted world points (P) and
    ground-truth world points (Q).

    GT points use normalized pixel coordinates [x/W, y/H, depth] to ensure a
    consistent coordinate space across all sequences and variants. Both sets
    are subsampled to n points before computing distances.

    Returns (Accuracy, Completeness) where:
      Accuracy     = mean nearest-neighbor distance from P to Q
      Completeness = mean nearest-neighbor distance from Q to P
      Overall      = (Accuracy + Completeness) / 2
    """
    pred_pts_list = []
    for fi in range(pred_world_points.shape[1]):
        wp = pred_world_points[0, fi].cpu().float().numpy().reshape(-1, 3)
        wc = pred_conf[0, fi].cpu().float().numpy().reshape(-1)
        pred_pts_list.append(wp[wc > 0.5])
    pred_pts = np.concatenate(pred_pts_list) if pred_pts_list else np.zeros((0, 3))

    gt_d_np    = gt_depth.squeeze(1).float().numpy()
    gt_mask_np = gt_mask.squeeze(1).numpy()
    gt_pts_list = []
    H, W = gt_d_np.shape[1], gt_d_np.shape[2]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for fi in range(gt_d_np.shape[0]):
        d = gt_d_np[fi]
        m = gt_mask_np[fi]
        pts = np.stack([xs[m].astype(np.float32) / W,
                        ys[m].astype(np.float32) / H,
                        d[m]], axis=-1)
        gt_pts_list.append(pts)
    gt_pts = np.concatenate(gt_pts_list) if gt_pts_list else np.zeros((0, 3))

    if len(pred_pts) < 10 or len(gt_pts) < 10:
        return float("nan"), float("nan")

    if len(pred_pts) > n:
        pred_pts = pred_pts[np.random.choice(len(pred_pts), n, replace=False)]
    if len(gt_pts) > n:
        gt_pts = gt_pts[np.random.choice(len(gt_pts), n, replace=False)]

    p   = torch.from_numpy(pred_pts).float().to(device)
    g   = torch.from_numpy(gt_pts).float().to(device)
    acc  = (p.unsqueeze(1) - g.unsqueeze(0)).norm(dim=-1).min(dim=1).values.mean().item()
    comp = (g.unsqueeze(1) - p.unsqueeze(0)).norm(dim=-1).min(dim=1).values.mean().item()
    return acc, comp


def _load_adapter_state(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ck)}")
    epoch = ck.get("epoch", None)
    if "adapter" in ck and isinstance(ck["adapter"], dict):
        return ck["adapter"], epoch
    if "adapters" in ck and isinstance(ck["adapters"], dict):
        return ck["adapters"], epoch
    if "model" in ck and isinstance(ck["model"], dict):
        full = ck["model"]
        adapter_state = {k[len("adapter."):]: v for k, v in full.items() if k.startswith("adapter.")}
        if adapter_state:
            return adapter_state, epoch
        return full, epoch
    if any(k.startswith("gate") or k.startswith("norm_q") or k.startswith("q_proj") for k in ck.keys()):
        return ck, epoch
    adapter_state = {k[len("adapter."):]: v for k, v in ck.items() if k.startswith("adapter.")}
    if adapter_state:
        return adapter_state, epoch
    raise ValueError(f"Could not identify adapter state. Keys: {list(ck.keys())[:10]}")


print("Loading CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
for p in clip_model.parameters():
    p.requires_grad_(False)
clip_model.eval()

print("Loading VGGT-1B probe...")
_probe = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
_probe.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        _t, _ = _probe.aggregator(torch.zeros(1, 2, 3, IMG_SIZE, IMG_SIZE, device=device))
        embed_dim = _t[-1].shape[-1] // 2
print(f"embed_dim={embed_dim}")

_probe_state_cpu = {k: v.cpu() for k, v in _probe.state_dict().items()}
_probe.cpu()
torch.cuda.empty_cache()
print("Probe weights cached on CPU, GPU freed")


def load_model(exp_name, model_module, model_class_name, ckpt_path, adapter_dim=512):
    if model_module is None:
        print(f"  {exp_name}: vanilla VGGT")
        m = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        m.eval()
        return m, False
    import importlib
    mod = importlib.import_module(model_module)
    cls = getattr(mod, model_class_name)
    m   = cls(
        clip_model=clip_model,
        img_size=IMG_SIZE,
        patch_size=14,
        embed_dim=embed_dim,
        adapter_dim=adapter_dim,
        d_text=512,
    ).to(device)
    m.load_state_dict(_probe_state_cpu, strict=False)
    adapter_state, epoch = _load_adapter_state(ckpt_path, device)
    if hasattr(m, "adapters"):
        m.adapters.load_state_dict(adapter_state)
        gate_val = m.adapters[0].gate.item()
    else:
        m.adapter.load_state_dict(adapter_state)
        gate_val = m.adapter.gate.item()
    print(f"  {exp_name}: loaded epoch={epoch} | gate={gate_val:.4f}")
    m.eval()
    return m, True


caption_lookup = {}
with open(CAPTION_FILE) as f:
    for line in f:
        r = json.loads(line)
        caption_lookup[r["source_seq_name"]] = r["caption_concise"]

sequences = load_sequences(ANNO_DIR, CO3D_DIR, split="test")
print(f"Test sequences: {len(sequences)}")


all_results = {}

for exp_name, exp_cfg in EXPERIMENTS.items():
    model_module, model_class_name, ckpt_path, adapter_dim = exp_cfg

    if ckpt_path is not None and not os.path.exists(ckpt_path):
        print(f"\n=== {exp_name} — checkpoint not found, skipping ===")
        continue

    print(f"\n=== {exp_name} ===")
    model, has_adapter = load_model(exp_name, model_module, model_class_name, ckpt_path, adapter_dim)
    metrics = []

    for i, seq in enumerate(sequences):
        caption = caption_lookup.get(seq["seq_name"], "")
        try:
            total   = len(seq["images"])
            n       = min(NUM_FRAMES, total)
            indices = np.linspace(0, total - 1, n, dtype=int)

            images            = load_images(seq["images"], indices).to(device)
            gt_depth, gt_mask = load_depths(seq["depths"], seq["masks"], indices)
            captions_arg      = [caption] if (has_adapter and caption) else None

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    pred = model(images, captions=captions_arg) if has_adapter \
                           else model(images)

            l1        = compute_l1(pred["depth"], gt_depth, gt_mask)
            acc, comp = compute_chamfer(
                pred["world_points"], pred["world_points_conf"],
                gt_depth, gt_mask,
            )

            metrics.append({
                "seq_name": seq["seq_name"],
                "category": seq["category"],
                "l1":       l1,
                "acc":      acc,
                "comp":     comp,
                "chamfer":  (acc + comp) / 2 if not (acc != acc or comp != comp) else float("nan"),
            })
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Skipped {seq['seq_name']}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sequences)} done")

    def nanmean(lst, k):
        vals = [m[k] for m in lst if not math.isnan(m.get(k, float("nan")))]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    summary = {
        "l1":          nanmean(metrics, "l1"),
        "acc":         nanmean(metrics, "acc"),
        "comp":        nanmean(metrics, "comp"),
        "chamfer":     nanmean(metrics, "chamfer"),
        "n_sequences": len(metrics),
    }
    all_results[exp_name] = {"summary": summary, "per_sequence": metrics}

    print(f"  L1={summary['l1']:.4f}  Acc={summary['acc']:.4f}  Comp={summary['comp']:.4f}  Overall={summary['chamfer']:.4f}  n={summary['n_sequences']}")

    del model
    torch.cuda.empty_cache()

with open(OUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved {OUT_FILE}")

def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "nan"

print(f"\n{'Experiment':<15} {'L1':>8} {'Acc':>8} {'Comp':>8} {'Overall':>10} {'N':>6}")
print("-" * 58)
for exp, data in all_results.items():
    s = data["summary"]
    print(f"{exp:<15} {_fmt(s['l1']):>8} {_fmt(s['acc']):>8} {_fmt(s['comp']):>8} {_fmt(s['chamfer']):>10} {s['n_sequences']:>6}")
