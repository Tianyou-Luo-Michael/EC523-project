"""
Launch script for FrozenVGGT training.

Usage (single GPU):
    python launch_frozen_vggt.py --config frozen_vggt

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 launch_frozen_vggt.py --config frozen_vggt

Override any YAML key on the command line (Hydra syntax):
    python launch_frozen_vggt.py --config frozen_vggt \
        caption_manifest_path=/data/captions.jsonl \
        adapter_dim=256 \
        max_epochs=50 \
        checkpoint.resume_checkpoint_path=/path/to/checkpoint.pt
"""

import argparse
from hydra import initialize, compose
from trainer_frozen_vggt import FrozenVGGTTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train FrozenVGGT (adapter-only) with a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="frozen_vggt",
        help="Config file name without .yaml extension (default: frozen_vggt)",
    )
    # Extra Hydra-style overrides: key=value pairs forwarded to OmegaConf.
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g.  adapter_dim=256  max_epochs=50",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    trainer = FrozenVGGTTrainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
