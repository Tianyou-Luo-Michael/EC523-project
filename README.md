# Text-Conditioned 3D Reconstruction with VGGT

**EC523 Deep Learning — Boston University**  
Timucin Erbas, Leroy Adisaputro, Azsadur Rakin, Tianyou Luo

This project extends **VGGT** with text conditioning for 3D reconstruction. The original VGGT backbone is kept frozen, and natural language captions are injected into the reconstruction pipeline through lightweight cross-attention adapters. The goal is to test whether semantic text information can improve depth prediction and point-cloud reconstruction without retraining the full VGGT model.

VGGT predicts camera pose, depth, and 3D point clouds from image sequences. Our method adds a frozen CLIP text encoder and a trainable `CrossAttentionAdapter` that conditions VGGT visual tokens on caption tokens before the prediction heads.

```text
Images  -> frozen VGGT aggregator -> visual tokens -+
Caption -> frozen CLIP encoder    -> text tokens   +-> adapter -> conditioned tokens -> VGGT heads
```

## File Overview

### Model files

Located in `vggt/models/`.

| File | Purpose |
|---|---|
| `adapter.py` | Defines `CrossAttentionAdapter`, the trainable module that fuses VGGT visual tokens with CLIP text-token embeddings. |
| `frozen_vggt.py` | Main frozen-VGGT model with one adapter injection at the final aggregator output. |
| `frozen_vggt_aggregator_0.py` | Variant that uses aggregator layer 0 for adapter conditioning. |
| `frozen_vggt_aggregator_3.py` | Variant that injects at multiple aggregator levels, including early, middle, and late representations. |
| `frozen_vggt_aggregator_12.py` | Variant that uses a middle aggregator layer for adapter conditioning. |
| `frozen_vggt_aggregator_dpt.py` | Variant that injects at the DPT feature-pyramid levels used by the depth/point heads. |
| `frozen_vggt_aggregator_everyother.py` | Variant that injects at every other aggregator level. |
| `frozen_vggt_multi_alldpt.py` | Variant with independent adapters at all DPT feature-pyramid levels. |

### Training files

Located in `training/`.

| File | Purpose |
|---|---|
| `trainer_frozen_vggt.py` | Base trainer for the frozen VGGT + adapter setup. Handles model setup, caption loading, optimizer setup, training steps, margin loss, checkpoint saving, and checkpoint loading. |
| `trainer_frozen_vggt_aggregator_0.py` | Trainer for the aggregator-0 injection experiment. |
| `trainer_frozen_vggt_aggregator_3.py` | Trainer for the multi-level aggregator injection experiment. |
| `trainer_frozen_vggt_aggregator_12.py` | Trainer for the aggregator-12 injection experiment. |
| `trainer_frozen_vggt_dpt.py` | Trainer for the DPT-level injection experiment. |
| `trainer_frozen_vggt_everyother.py` | Trainer for the every-other-layer injection experiment. |
| `trainer_frozen_vggt_multi_alldpt.py` | Trainer for the independent multi-DPT-adapter experiment. |
| `launch_frozen_vggt.py` | Launch script for the main frozen-VGGT adapter experiment. |
| `launch_frozen_vggt_aggregator_0.py` | Launch script for the aggregator-0 experiment. |
| `launch_frozen_vggt_aggregator_3.py` | Launch script for the multi-level aggregator experiment. |
| `launch_frozen_vggt_aggregator_12.py` | Launch script for the aggregator-12 experiment. |
| `launch_frozen_vggt_dim128.py` | Launch script for the adapter-dimension 128 ablation. |
| `launch_frozen_vggt_dim256.py` | Launch script for the adapter-dimension 256 ablation. |
| `launch_frozen_vggt_dim1024.py` | Launch script for the adapter-dimension 1024 ablation. |
| `launch_frozen_vggt_dpt.py` | Launch script for the DPT-level injection experiment. |
| `launch_frozen_vggt_everyother.py` | Launch script for the every-other-layer experiment. |
| `launch_frozen_vggt_multi_alldpt.py` | Launch script for the independent multi-DPT-adapter experiment. |

### Configuration files

Located in `training/config/`.

| File | Purpose |
|---|---|
| `frozen_vggt.yaml` | Main training configuration for the base frozen-VGGT adapter experiment. |
| `frozen_vggt_aggregator_0.yaml` | Configuration for the aggregator-0 experiment. |
| `frozen_vggt_aggregator_3.yaml` | Configuration for the multi-level aggregator experiment. |
| `frozen_vggt_aggregator_12.yaml` | Configuration for the aggregator-12 experiment. |
| `frozen_vggt_dim128.yaml` | Configuration for the adapter-dimension 128 ablation. |
| `frozen_vggt_dim256.yaml` | Configuration for the adapter-dimension 256 ablation. |
| `frozen_vggt_dim1024.yaml` | Configuration for the adapter-dimension 1024 ablation. |
| `frozen_vggt_dpt.yaml` | Configuration for the DPT-level injection experiment. |
| `frozen_vggt_everyother.yaml` | Configuration for the every-other-layer injection experiment. |
| `frozen_vggt_multi_alldpt.yaml` | Configuration for the independent multi-DPT-adapter experiment. |

### Evaluation and submission files

Located at the project root unless otherwise noted.

| File | Purpose |
|---|---|
| `evaluation.py` | Evaluates model variants on the Co3D test set and reports L1 depth error and Chamfer distance metrics. |
| `submit_aggregator_0.sh` | BU SCC SGE submission script for the aggregator-0 experiment. |
| `submit_aggregator_3.sh` | BU SCC SGE submission script for the multi-level aggregator experiment. |
| `submit_aggregator_12.sh` | BU SCC SGE submission script for the aggregator-12 experiment. |
| `submit_dim128.sh` | BU SCC SGE submission script for the adapter-dimension 128 ablation. |
| `submit_dim256.sh` | BU SCC SGE submission script for the adapter-dimension 256 ablation. |
| `submit_dim1024.sh` | BU SCC SGE submission script for the adapter-dimension 1024 ablation. |
| `submit_dpt.sh` | BU SCC SGE submission script for the DPT-level injection experiment. |
| `submit_everyother.sh` | BU SCC SGE submission script for the every-other-layer experiment. |
| `submit_multi_alldpt.sh` | BU SCC SGE submission script for the independent multi-DPT-adapter experiment. |

## Installation

Clone the repository:

```bash
git clone https://github.com/Tianyou-Luo-Michael/EC523-project.git
cd EC523-project
```

Create and activate a Python environment:

```bash
python -m venv env
source env/bin/activate
```

On Windows PowerShell, use:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

The project requires a CUDA-capable GPU for training. The experiments were designed for large-memory GPUs. Some variants may require 40GB to 80GB of GPU memory depending on the injection strategy and batch size.

## Dataset Setup

The experiments use the Co3D dataset and caption files. Before training or evaluation, update the dataset paths in the relevant YAML config files under:

```text
training/config/
```

Typical paths to check include:

```text
CO3D image/data directory
annotation split directory
caption manifest path
checkpoint save directory
logging directory
```

The exact path names depend on the local machine or cluster environment.

## Training

Run a training experiment with `torchrun`. For a single-GPU run:

```bash
torchrun --nproc_per_node=1 training/launch_frozen_vggt.py
```

Examples for specific variants:

```bash
torchrun --nproc_per_node=1 training/launch_frozen_vggt_aggregator_0.py
torchrun --nproc_per_node=1 training/launch_frozen_vggt_aggregator_12.py
torchrun --nproc_per_node=1 training/launch_frozen_vggt_aggregator_3.py
torchrun --nproc_per_node=1 training/launch_frozen_vggt_dpt.py
torchrun --nproc_per_node=1 training/launch_frozen_vggt_everyother.py
torchrun --nproc_per_node=1 training/launch_frozen_vggt_multi_alldpt.py
```

Adapter-dimension ablations:

```bash
torchrun --nproc_per_node=1 training/launch_frozen_vggt_dim128.py
torchrun --nproc_per_node=1 training/launch_frozen_vggt_dim256.py
torchrun --nproc_per_node=1 training/launch_frozen_vggt_dim1024.py
```

Hydra-style overrides can be passed from the command line. Example:

```bash
torchrun --nproc_per_node=1 training/launch_frozen_vggt_dpt.py max_epochs=20 adapter_dim=512
```

Only the adapter parameters are trained. VGGT and CLIP remain frozen.

## Running on the BU SCC Cluster

The `submit_*.sh` files are SGE batch scripts for the BU Shared Computing Cluster. To submit one experiment:

```bash
qsub submit_dpt.sh
```

Other examples:

```bash
qsub submit_aggregator_0.sh
qsub submit_aggregator_12.sh
qsub submit_aggregator_3.sh
qsub submit_dim128.sh
qsub submit_dim256.sh
qsub submit_dim1024.sh
qsub submit_everyother.sh
qsub submit_multi_alldpt.sh
```

Before submitting, check that each script points to the correct project directory, Python environment, launch script, GPU type, GPU memory requirement, and log path.

## Evaluation

Before running evaluation, update the dataset and checkpoint paths inside `evaluation.py`.

Important variables to edit:

```python
CO3D_DIR = "/path/to/co3d_curated"
ANNO_DIR = "/path/to/co3d_curated_anno_split"
CAPTION_FILE = "/path/to/co3d_curated_captions_test.jsonl"
OUT_FILE = "logs/eval_results.json"
```

Also update the checkpoint path for each experiment in the `EXPERIMENTS` dictionary.

Then run:

```bash
python evaluation.py
```

The script reports:

```text
L1 depth error
Chamfer Accuracy
Chamfer Completeness
Chamfer Overall
number of evaluated sequences
```

The full results are saved to the JSON file specified by `OUT_FILE`.

## Reproducing Results

To reproduce the experiments:

1. Clone the repository.
2. Install the required Python packages.
3. Download and prepare Co3D data and caption files.
4. Update paths in the YAML config files.
5. Train the desired model variants using the launch scripts or SCC submit scripts.
6. Update checkpoint paths in `evaluation.py`.
7. Run `python evaluation.py` to generate the final metrics table.

## Video Demos

Add video demos here if available:

```text
Demo 1: <link or filename>
Demo 2: <link or filename>
```

## Citation

This project builds on VGGT:

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={CVPR},
  year={2025}
}
```
