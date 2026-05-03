import argparse
from hydra import initialize, compose
from trainer_frozen_vggt import FrozenVGGTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="frozen_vggt_dim1024")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)
    FrozenVGGTTrainer(**cfg).run()


if __name__ == "__main__":
    main()
