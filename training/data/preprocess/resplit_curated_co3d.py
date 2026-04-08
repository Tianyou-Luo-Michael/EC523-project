"""
Resplit a curated Co3D subset into VGGT-compatible train/test annotations.

This script is intended for the case where a curated Co3D subset already exists
on disk, but the current curated annotations only define a train split.

It does four things:
1. Loads the existing curated annotation entries and validates that the dataset
   files required by VGGT actually exist for each sequence.
2. Derives a per-category train/test ratio from the original Co3D annotation
   files rather than using a naive 50/50 split.
3. Writes matching ``<category>_train.jgz`` and ``<category>_test.jgz`` files
   for the curated subset.
4. Splits an existing caption JSONL manifest into train/test manifests using the
   same sequence assignment, while rewriting ``sampled_image_paths`` to match the
   dataset root used for training.

Typical SCC usage:
    python training/data/preprocess/resplit_curated_co3d.py \
        --curated-co3d-root /projectnb/ec523/projects/proj_vggt/dataset/co3d_curated \
        --input-annotation-dir /projectnb/ec523/projects/proj_vggt/dataset/co3d_curated_anno \
        --source-annotation-dir /projectnb/ec523/students/tianyoul/data/co3d_anno \
        --input-caption-manifest /projectnb/ec523/students/tianyoul/data/co3d_curated_captions.jsonl \
        --output-annotation-dir /projectnb/ec523/projects/proj_vggt/dataset/co3d_curated_anno_split \
        --output-caption-train /projectnb/ec523/projects/proj_vggt/dataset/co3d_curated_captions_train.jsonl \
        --output-caption-test /projectnb/ec523/projects/proj_vggt/dataset/co3d_curated_captions_test.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable


@dataclass
class CategorySummary:
    category: str
    available_sequences: int
    invalid_sequences: int
    original_train_sequences: int
    original_test_sequences: int
    derived_train_ratio: float
    train_sequences_written: int
    test_sequences_written: int
    captions_train_written: int
    captions_test_written: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curated-co3d-root",
        type=Path,
        required=True,
        help="Curated Co3D dataset root containing category/sequence directories.",
    )
    parser.add_argument(
        "--input-annotation-dir",
        type=Path,
        required=True,
        help="Existing curated annotation directory to resplit.",
    )
    parser.add_argument(
        "--source-annotation-dir",
        type=Path,
        required=True,
        help="Original Co3D annotation directory used to derive train/test proportions.",
    )
    parser.add_argument(
        "--output-annotation-dir",
        type=Path,
        default=None,
        help=(
            "Output annotation directory for the new train/test split. "
            "Defaults to <input-annotation-dir>_resplit."
        ),
    )
    parser.add_argument(
        "--input-caption-manifest",
        type=Path,
        default=None,
        help="Optional caption JSONL manifest to split alongside the annotations.",
    )
    parser.add_argument(
        "--output-caption-train",
        type=Path,
        default=None,
        help="Optional output train caption JSONL path. Defaults beside the input manifest.",
    )
    parser.add_argument(
        "--output-caption-test",
        type=Path,
        default=None,
        help="Optional output test caption JSONL path. Defaults beside the input manifest.",
    )
    parser.add_argument(
        "--caption-co3d-root",
        type=Path,
        default=None,
        help=(
            "Dataset root to encode into output caption sampled_image_paths. "
            "Defaults to --curated-co3d-root."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category subset. Defaults to categories discovered in the input annotations.",
    )
    parser.add_argument(
        "--input-splits",
        nargs="*",
        default=["train", "test"],
        help="Existing curated splits to union before resplitting. Missing files are skipped.",
    )
    parser.add_argument(
        "--min-num-images",
        type=int,
        default=24,
        help="Minimum sequence length, matching VGGT's Co3D loader behavior.",
    )
    parser.add_argument(
        "--default-train-ratio",
        type=float,
        default=0.75,
        help="Fallback train ratio if the original annotations are unavailable for a category.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-captions",
        action="store_true",
        help="Allow output annotation entries without corresponding caption records.",
    )
    parser.add_argument(
        "--skip-depth-check",
        action="store_true",
        help="Skip depth/depth-mask existence checks. Not recommended for VGGT training.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report path. Defaults inside the output annotation directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    if args.output_annotation_dir is None:
        args.output_annotation_dir = args.input_annotation_dir.parent / (
            args.input_annotation_dir.name + "_resplit"
        )

    if args.caption_co3d_root is None:
        args.caption_co3d_root = args.curated_co3d_root

    if args.input_caption_manifest is not None:
        if args.output_caption_train is None:
            args.output_caption_train = args.input_caption_manifest.with_name(
                f"{args.input_caption_manifest.stem}_train{args.input_caption_manifest.suffix}"
            )
        if args.output_caption_test is None:
            args.output_caption_test = args.input_caption_manifest.with_name(
                f"{args.input_caption_manifest.stem}_test{args.input_caption_manifest.suffix}"
            )

    if args.input_caption_manifest is None and (
        args.output_caption_train is not None or args.output_caption_test is not None
    ):
        parser.error(
            "--output-caption-train/--output-caption-test require --input-caption-manifest."
        )

    if args.input_caption_manifest is not None and (
        args.output_caption_train is None or args.output_caption_test is None
    ):
        parser.error(
            "Both output caption paths must be defined when --input-caption-manifest is used."
        )

    return args


def load_annotation_file(annotation_file: Path) -> Dict[str, list]:
    with gzip.open(annotation_file, "rt", encoding="utf-8") as handle:
        return json.loads(handle.read())


def write_annotation_file(annotation_file: Path, payload: Dict[str, list]) -> None:
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(annotation_file, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def iter_caption_records(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {manifest_path}: {exc}"
                ) from exc


def load_caption_records(manifest_path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for record in iter_caption_records(manifest_path):
        key = record.get("dataset_seq_name")
        if key is None:
            raise KeyError(
                f"Caption record missing required 'dataset_seq_name': {record}"
            )
        if key in records:
            raise ValueError(
                f"Duplicate caption record key '{key}' found in {manifest_path}"
            )
        records[key] = record
    return records


def write_jsonl(output_path: Path, records: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_sidecar_metadata(manifest_path: Path) -> dict | None:
    metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_caption_metadata(
    output_path: Path,
    source_manifest: Path,
    split_name: str,
    annotation_dir: Path,
    co3d_root: Path,
    records: list[dict],
    source_metadata: dict | None,
) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    selected_categories = sorted({record["category"] for record in records})
    metadata = {
        "script": "resplit_curated_co3d.py",
        "source_manifest": str(source_manifest),
        "split": split_name,
        "input_mode": "annotations",
        "co3d_dir": str(co3d_root),
        "co3d_annotation_dir": str(annotation_dir),
        "categories": selected_categories,
        "total_sequences_selected": len(records),
        "total_sequences_eligible_before_sampling": len(records),
    }

    if source_metadata is not None:
        for field in (
            "model_name",
            "min_num_images",
            "frames_per_sequence",
            "max_sequences",
            "max_sequences_per_category",
            "shuffle",
            "seed",
            "caption_format",
            "use_category_hint",
            "prompt",
        ):
            if field in source_metadata:
                metadata[field] = source_metadata[field]

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def discover_categories(
    input_annotation_dir: Path,
    curated_co3d_root: Path,
    input_splits: list[str],
) -> list[str]:
    categories = set()
    for split_name in input_splits:
        suffix = f"_{split_name}.jgz"
        for annotation_file in input_annotation_dir.glob(f"*{suffix}"):
            categories.add(annotation_file.name[: -len(suffix)])

    if not categories and curated_co3d_root.exists():
        categories.update(path.name for path in curated_co3d_root.iterdir() if path.is_dir())

    return sorted(categories)


def load_annotation_union(
    annotation_dir: Path,
    category: str,
    split_names: list[str],
) -> Dict[str, list]:
    merged: Dict[str, list] = {}
    for split_name in split_names:
        annotation_file = annotation_dir / f"{category}_{split_name}.jgz"
        if not annotation_file.exists():
            continue
        payload = load_annotation_file(annotation_file)
        for seq_name, seq_data in payload.items():
            merged[seq_name] = seq_data
    return merged


def load_eligible_source_sets(
    source_annotation_dir: Path,
    category: str,
    min_num_images: int,
) -> tuple[set[str], set[str]]:
    eligible_train: set[str] = set()
    eligible_test: set[str] = set()
    for split_name, destination in (("train", eligible_train), ("test", eligible_test)):
        annotation_file = source_annotation_dir / f"{category}_{split_name}.jgz"
        if not annotation_file.exists():
            continue
        payload = load_annotation_file(annotation_file)
        for seq_name, seq_data in payload.items():
            if len(seq_data) >= min_num_images:
                destination.add(seq_name)
    return eligible_train, eligible_test


def replace_path_component(path: Path, old_component: str, new_component: str) -> Path:
    parts = list(path.parts)
    try:
        index = parts.index(old_component)
    except ValueError as exc:
        raise ValueError(f"Expected path component '{old_component}' in {path}") from exc
    parts[index] = new_component
    return Path(*parts)


def validate_sequence(
    curated_co3d_root: Path,
    seq_data: list,
    skip_depth_check: bool,
) -> tuple[bool, str | None]:
    for anno in seq_data:
        relative_image_path = Path(anno["filepath"])
        image_path = curated_co3d_root / relative_image_path
        if not image_path.exists():
            return False, f"missing image {image_path}"

        if skip_depth_check:
            continue

        relative_depth_path = replace_path_component(
            relative_image_path, "images", "depths"
        )
        depth_path = curated_co3d_root / Path(str(relative_depth_path) + ".geometric.png")
        if not depth_path.exists():
            return False, f"missing depth {depth_path}"

        relative_mask_path = replace_path_component(
            relative_image_path, "images", "depth_masks"
        ).with_suffix(".png")
        mask_path = curated_co3d_root / relative_mask_path
        if not mask_path.exists():
            return False, f"missing depth mask {mask_path}"

    return True, None


def compute_desired_train_count(
    available_count: int,
    original_train_count: int,
    original_test_count: int,
    default_train_ratio: float,
) -> tuple[int, float]:
    original_total = original_train_count + original_test_count
    if original_total > 0:
        train_ratio = original_train_count / original_total
    else:
        train_ratio = default_train_ratio

    desired_train = round(available_count * train_ratio)
    if available_count >= 2:
        desired_train = max(1, min(available_count - 1, desired_train))
    else:
        desired_train = available_count
    return desired_train, train_ratio


def take_sequences(
    desired_count: int,
    priority_groups: list[list[str]],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for group in priority_groups:
        for seq_name in group:
            if seq_name in seen:
                continue
            selected.append(seq_name)
            seen.add(seq_name)
            if len(selected) >= desired_count:
                return selected
    return selected


def assign_category_splits(
    seq_names: list[str],
    original_train: set[str],
    original_test: set[str],
    desired_train_count: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    train_only = [seq for seq in seq_names if seq in original_train and seq not in original_test]
    test_only = [seq for seq in seq_names if seq in original_test and seq not in original_train]
    both = [seq for seq in seq_names if seq in original_train and seq in original_test]
    neither = [seq for seq in seq_names if seq not in original_train and seq not in original_test]

    for group in (train_only, test_only, both, neither):
        rng.shuffle(group)

    desired_test_count = len(seq_names) - desired_train_count
    test_selected = take_sequences(
        desired_test_count,
        [test_only, both, neither, train_only],
    )
    test_set = set(test_selected)

    train_candidates = [
        [seq for seq in group if seq not in test_set]
        for group in (train_only, both, neither, test_only)
    ]
    train_selected = take_sequences(desired_train_count, train_candidates)

    selected_union = set(train_selected) | test_set
    unassigned = [seq for seq in seq_names if seq not in selected_union]
    train_selected.extend(unassigned)

    if len(train_selected) + len(test_selected) != len(seq_names):
        raise RuntimeError("Failed to assign every sequence to exactly one split.")

    return sorted(train_selected), sorted(test_selected)


def rewrite_sampled_image_paths(
    record: dict,
    co3d_root: Path,
) -> list[str]:
    category = record["category"]
    sequence_dir_name = Path(record["source_seq_name"]).name
    rewritten = []
    for path_str in record.get("sampled_image_paths", []):
        filename = Path(path_str).name
        rewritten.append(
            str((co3d_root / category / sequence_dir_name / "images" / filename).resolve())
        )
    return rewritten


def ensure_output_path_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {path}. Use --overwrite to replace it."
        )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    categories = sorted(args.categories or discover_categories(
        input_annotation_dir=args.input_annotation_dir,
        curated_co3d_root=args.curated_co3d_root,
        input_splits=args.input_splits,
    ))

    if not categories:
        raise ValueError("No categories found to resplit.")

    if args.output_annotation_dir.resolve() == args.input_annotation_dir.resolve() and not args.overwrite:
        raise ValueError(
            "Refusing to overwrite the input annotation directory in place without --overwrite."
        )

    if args.input_caption_manifest is not None:
        ensure_output_path_available(args.output_caption_train, args.overwrite)
        ensure_output_path_available(args.output_caption_test, args.overwrite)

    args.output_annotation_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_path or (args.output_annotation_dir / "resplit_report.json")
    ensure_output_path_available(report_path, args.overwrite)

    caption_records = None
    caption_metadata = None
    if args.input_caption_manifest is not None:
        caption_records = load_caption_records(args.input_caption_manifest)
        caption_metadata = load_sidecar_metadata(args.input_caption_manifest)

    split_lookup: dict[str, str] = {}
    category_summaries: list[CategorySummary] = []
    invalid_details: dict[str, dict[str, str]] = {}

    for category in categories:
        category_annotation = load_annotation_union(
            annotation_dir=args.input_annotation_dir,
            category=category,
            split_names=args.input_splits,
        )
        if not category_annotation:
            print(f"[warn] No curated annotation entries found for category {category}, skipping.")
            continue

        valid_entries: Dict[str, list] = {}
        invalid_reasons: dict[str, str] = {}
        for seq_name, seq_data in sorted(category_annotation.items()):
            if len(seq_data) < args.min_num_images:
                invalid_reasons[seq_name] = (
                    f"sequence shorter than min_num_images={args.min_num_images}"
                )
                continue
            is_valid, reason = validate_sequence(
                curated_co3d_root=args.curated_co3d_root,
                seq_data=seq_data,
                skip_depth_check=args.skip_depth_check,
            )
            if not is_valid:
                invalid_reasons[seq_name] = reason or "unknown validation failure"
                continue
            valid_entries[seq_name] = seq_data

        invalid_details[category] = invalid_reasons
        available_seq_names = sorted(valid_entries)
        if not available_seq_names:
            print(f"[warn] No valid curated sequences remain for category {category}, skipping.")
            continue

        original_train, original_test = load_eligible_source_sets(
            source_annotation_dir=args.source_annotation_dir,
            category=category,
            min_num_images=args.min_num_images,
        )
        desired_train_count, train_ratio = compute_desired_train_count(
            available_count=len(available_seq_names),
            original_train_count=len(original_train),
            original_test_count=len(original_test),
            default_train_ratio=args.default_train_ratio,
        )
        train_seq_names, test_seq_names = assign_category_splits(
            seq_names=available_seq_names,
            original_train=original_train,
            original_test=original_test,
            desired_train_count=desired_train_count,
            rng=rng,
        )

        train_payload = {seq_name: valid_entries[seq_name] for seq_name in train_seq_names}
        test_payload = {seq_name: valid_entries[seq_name] for seq_name in test_seq_names}

        write_annotation_file(
            args.output_annotation_dir / f"{category}_train.jgz",
            train_payload,
        )
        write_annotation_file(
            args.output_annotation_dir / f"{category}_test.jgz",
            test_payload,
        )

        for seq_name in train_seq_names:
            split_lookup[f"co3d_{seq_name}"] = "train"
        for seq_name in test_seq_names:
            split_lookup[f"co3d_{seq_name}"] = "test"

        category_summaries.append(
            CategorySummary(
                category=category,
                available_sequences=len(available_seq_names),
                invalid_sequences=len(invalid_reasons),
                original_train_sequences=len(original_train),
                original_test_sequences=len(original_test),
                derived_train_ratio=train_ratio,
                train_sequences_written=len(train_seq_names),
                test_sequences_written=len(test_seq_names),
                captions_train_written=0,
                captions_test_written=0,
            )
        )
        print(
            f"[split] {category}: {len(train_seq_names)} train / {len(test_seq_names)} test "
            f"(ratio {train_ratio:.3f}, invalid dropped {len(invalid_reasons)})"
        )

    missing_caption_keys: list[str] = []
    train_caption_records: list[dict] = []
    test_caption_records: list[dict] = []

    if caption_records is not None:
        for dataset_seq_name, split_name in split_lookup.items():
            record = caption_records.get(dataset_seq_name)
            if record is None:
                missing_caption_keys.append(dataset_seq_name)
                continue

            updated_record = dict(record)
            updated_record["split"] = split_name
            updated_record["sampled_image_paths"] = rewrite_sampled_image_paths(
                record=record,
                co3d_root=args.caption_co3d_root,
            )

            if split_name == "train":
                train_caption_records.append(updated_record)
            elif split_name == "test":
                test_caption_records.append(updated_record)
            else:
                raise RuntimeError(f"Unexpected split name: {split_name}")

        if missing_caption_keys and not args.allow_missing_captions:
            raise ValueError(
                "Missing caption records for selected sequences. "
                f"Examples: {missing_caption_keys[:10]}"
            )

        write_jsonl(args.output_caption_train, train_caption_records)
        write_jsonl(args.output_caption_test, test_caption_records)
        write_caption_metadata(
            output_path=args.output_caption_train,
            source_manifest=args.input_caption_manifest,
            split_name="train",
            annotation_dir=args.output_annotation_dir,
            co3d_root=args.caption_co3d_root,
            records=train_caption_records,
            source_metadata=caption_metadata,
        )
        write_caption_metadata(
            output_path=args.output_caption_test,
            source_manifest=args.input_caption_manifest,
            split_name="test",
            annotation_dir=args.output_annotation_dir,
            co3d_root=args.caption_co3d_root,
            records=test_caption_records,
            source_metadata=caption_metadata,
        )

        summary_by_category = {summary.category: summary for summary in category_summaries}
        for record in train_caption_records:
            summary_by_category[record["category"]].captions_train_written += 1
        for record in test_caption_records:
            summary_by_category[record["category"]].captions_test_written += 1

    report = {
        "script": "resplit_curated_co3d.py",
        "curated_co3d_root": str(args.curated_co3d_root),
        "input_annotation_dir": str(args.input_annotation_dir),
        "output_annotation_dir": str(args.output_annotation_dir),
        "source_annotation_dir": str(args.source_annotation_dir),
        "input_splits": list(args.input_splits),
        "categories": categories,
        "min_num_images": args.min_num_images,
        "default_train_ratio": args.default_train_ratio,
        "skip_depth_check": args.skip_depth_check,
        "input_caption_manifest": (
            str(args.input_caption_manifest) if args.input_caption_manifest is not None else None
        ),
        "output_caption_train": (
            str(args.output_caption_train) if args.output_caption_train is not None else None
        ),
        "output_caption_test": (
            str(args.output_caption_test) if args.output_caption_test is not None else None
        ),
        "missing_caption_keys": missing_caption_keys,
        "category_summaries": [asdict(summary) for summary in category_summaries],
        "invalid_sequences": invalid_details,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[summary] Report written to: {report_path}")


if __name__ == "__main__":
    main()
