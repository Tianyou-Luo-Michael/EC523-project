"""
Launch script for VGGTTextOnly training.

Usage (single GPU):
    python launch_vggt_textonly.py --config vggt_textonly

Usage (multi-GPU, e.g. 4 GPUs):
    torchrun --nproc_per_node=4 launch_vggt_textonly.py --config vggt_textonly

Override any YAML key on the command line (Hydra syntax):
    python launch_vggt_textonly.py --config vggt_textonly \\
        caption_manifest_path=/data/captions.jsonl \\
        alignment_weight=2.0 \\
        task_weight=0.0 \\
        freeze_heads_steps=1000 \\
        max_epochs=50 \\
        checkpoint.resume_checkpoint_path=/path/to/checkpoint.pt
"""

import argparse
from hydra import initialize, compose
from trainer_vggt_textonly import VGGTTextOnlyTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train VGGTTextOnly (text encoder alignment) with a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vggt_textonly",
        help="Config file name without .yaml extension (default: vggt_textonly)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g.  alignment_weight=2.0  task_weight=0.0",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    trainer = VGGTTextOnlyTrainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
