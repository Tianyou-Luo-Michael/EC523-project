"""
Run VGGTTextOnly evaluation on a dataset split (default: test).

Examples:
    python test_vggt_textonly.py --config vggt_textonly
    python test_vggt_textonly.py --config vggt_textonly --checkpoint /path/to/ckpt.pt
    torchrun --nproc_per_node=4 test_vggt_textonly.py --config vggt_textonly
"""

import argparse
import os

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from trainer_vggt_textonly import VGGTTextOnlyTrainer


def _override_eval_split(cfg: DictConfig, split: str) -> None:
    """Set split for each configured validation dataset entry."""
    val_cfg = cfg.get("data", {}).get("val", None)
    if val_cfg is None:
        raise ValueError("Config is missing data.val. Cannot run evaluation.")

    dataset_cfg = val_cfg.get("dataset", None)
    if dataset_cfg is None:
        raise ValueError("Config is missing data.val.dataset. Cannot set split.")

    dataset_configs = dataset_cfg.get("dataset_configs", None)
    if not dataset_configs:
        raise ValueError("Config is missing data.val.dataset.dataset_configs.")

    updated = 0
    for ds in dataset_configs:
        if "split" in ds:
            ds.split = split
            updated += 1

    if updated == 0:
        raise ValueError(
            "No 'split' field found under data.val.dataset.dataset_configs entries."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VGGTTextOnly on a target dataset split."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vggt_textonly",
        help="Config name in training/config (without .yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path for text-encoder/head weights.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Optional cap on number of eval batches for quick smoke runs.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides, e.g. logging.log_freq=1.",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    cfg.mode = "val"
    _override_eval_split(cfg, args.split)

    if args.checkpoint is not None:
        cfg.checkpoint.resume_checkpoint_path = args.checkpoint

    if args.limit_val_batches is not None:
        cfg.limit_val_batches = args.limit_val_batches

    print("===== Evaluation config (resolved) =====")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = VGGTTextOnlyTrainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
