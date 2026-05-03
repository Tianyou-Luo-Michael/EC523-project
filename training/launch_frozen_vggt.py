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
