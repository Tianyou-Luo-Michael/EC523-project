import argparse
from hydra import initialize, compose
from trainer_frozen_vggt_aggregator_3 import FrozenVGGTTrainer_Aggregator_3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="frozen_vggt_aggregator_3")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    FrozenVGGTTrainer_Aggregator_3(**cfg).run()


if __name__ == "__main__":
    main()
