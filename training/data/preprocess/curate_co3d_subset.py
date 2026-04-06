"""
Curate a smaller, trainable Co3D subset for VGGT-style training.

This script is designed for the "download one category chunk to scratch, keep only
some sequences, then delete the temporary download" workflow.

It does three important things:
1. Copies the full directory for each selected sequence into a persistent curated root.
2. Writes filtered ``<category>_<split>.jgz`` annotation files containing only the
   sequences that were kept.
3. Stops adding sequences once the curated dataset hits a storage budget.

Typical usage on SCC:
    python training/data/preprocess/curate_co3d_subset.py \
        --source-co3d-root /scratch/$USER/co3d_wave \
        --source-annotation-dir /projectnb/ec523/students/tianyoul/data/co3d_anno \
        --curated-co3d-root /projectnb/ec523/students/tianyoul/data/co3d_curated \
        --curated-annotation-dir /projectnb/ec523/students/tianyoul/data/co3d_curated_anno \
        --split train \
        --categories apple \
        --target-sequences-per-category 20 \
        --max-curated-gb 100 \
        --shuffle

Run the script repeatedly after each temporary chunk download. It will resume from
existing curated data and only add more sequences when a category is still below
its target count.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


DEFAULT_SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]


@dataclass
class CategoryResult:
    category: str
    existing_sequences: int
    added_sequences: int
    target_sequences: int
    size_limit_reached: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-co3d-root",
        type=Path,
        required=True,
        help="Temporary Co3D root containing currently downloaded category data.",
    )
    parser.add_argument(
        "--source-annotation-dir",
        type=Path,
        required=True,
        help="Directory containing the original <category>_<split>.jgz files.",
    )
    parser.add_argument(
        "--curated-co3d-root",
        type=Path,
        required=True,
        help="Persistent root where selected sequence folders will be copied.",
    )
    parser.add_argument(
        "--curated-annotation-dir",
        type=Path,
        required=True,
        help="Directory where filtered <category>_<split>.jgz files will be written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Output split name for the curated annotation files.",
    )
    parser.add_argument(
        "--source-splits",
        nargs="*",
        default=None,
        help=(
            "Source annotation splits to union when selecting sequences. "
            "Defaults to the output split only. Example: --source-splits train test"
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category subset. Defaults to the VGGT seen categories.",
    )
    parser.add_argument(
        "--target-sequences-per-category",
        type=int,
        default=20,
        help="Persistent target count for each category across repeated runs.",
    )
    parser.add_argument(
        "--min-num-images",
        type=int,
        default=24,
        help="Skip sequences shorter than this, matching VGGT defaults.",
    )
    parser.add_argument(
        "--max-curated-gb",
        type=float,
        default=100.0,
        help="Hard cap for the curated dataset size.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle candidate sequence order before selecting new sequences.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report summarizing what this run added.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be added without copying files or writing annotations.",
    )
    return parser.parse_args()


def load_annotation_file(annotation_file: Path) -> Dict[str, list]:
    with gzip.open(annotation_file, "rt", encoding="utf-8") as handle:
        return json.loads(handle.read())


def write_annotation_file(annotation_file: Path, payload: Dict[str, list]) -> None:
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(annotation_file, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_category_annotation_union(
    source_annotation_dir: Path,
    category: str,
    source_splits: list[str],
) -> Dict[str, list]:
    merged: Dict[str, list] = {}
    for split_name in source_splits:
        annotation_file = source_annotation_dir / f"{category}_{split_name}.jgz"
        if not annotation_file.exists():
            print(f"[warn] Missing source annotation file, skipping: {annotation_file}")
            continue
        payload = load_annotation_file(annotation_file)
        duplicate_count = 0
        for seq_name, seq_data in payload.items():
            if seq_name in merged:
                duplicate_count += 1
            merged[seq_name] = seq_data
        if duplicate_count:
            print(
                f"[warn] {category}: {duplicate_count} duplicate sequence ids encountered "
                f"while merging split {split_name}."
            )
    return merged


def directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def format_size_gb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def resolve_annotation_seq_name(
    category: str,
    sequence_dir_name: str,
    category_annotation: Dict[str, list],
) -> str | None:
    candidates = [sequence_dir_name, f"{category}/{sequence_dir_name}"]
    for candidate in candidates:
        if candidate in category_annotation:
            return candidate
    return None


def sync_existing_annotation_entries(
    category: str,
    category_annotation: Dict[str, list],
    curated_category_dir: Path,
    selected_entries: Dict[str, list],
    min_num_images: int,
) -> int:
    recovered = 0
    if not curated_category_dir.is_dir():
        return recovered

    for sequence_dir in sorted(path for path in curated_category_dir.iterdir() if path.is_dir()):
        seq_name = resolve_annotation_seq_name(
            category=category,
            sequence_dir_name=sequence_dir.name,
            category_annotation=category_annotation,
        )
        if seq_name is None:
            continue
        if seq_name in selected_entries:
            continue
        seq_data = category_annotation.get(seq_name)
        if seq_data is None or len(seq_data) < min_num_images:
            continue
        selected_entries[seq_name] = seq_data
        recovered += 1
    return recovered


def collect_available_candidates(
    category: str,
    category_annotation: Dict[str, list],
    source_category_dir: Path,
    selected_entries: Dict[str, list],
    min_num_images: int,
) -> list[tuple[str, Path, list]]:
    candidates: list[tuple[str, Path, list]] = []
    if not source_category_dir.is_dir():
        return candidates

    for sequence_dir in sorted(path for path in source_category_dir.iterdir() if path.is_dir()):
        images_dir = sequence_dir / "images"
        if not images_dir.is_dir():
            continue
        seq_name = resolve_annotation_seq_name(
            category=category,
            sequence_dir_name=sequence_dir.name,
            category_annotation=category_annotation,
        )
        if seq_name is None:
            continue
        if seq_name in selected_entries:
            continue
        seq_data = category_annotation.get(seq_name)
        if seq_data is None or len(seq_data) < min_num_images:
            continue
        candidates.append((seq_name, sequence_dir, seq_data))
    return candidates


def copy_sequence_dir(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, dest_dir)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    source_splits = list(args.source_splits) if args.source_splits else [args.split]

    categories = sorted(args.categories or DEFAULT_SEEN_CATEGORIES)
    args.curated_co3d_root.mkdir(parents=True, exist_ok=True)
    args.curated_annotation_dir.mkdir(parents=True, exist_ok=True)

    max_curated_bytes = int(args.max_curated_gb * (1024 ** 3))
    curated_size_bytes = directory_size_bytes(args.curated_co3d_root)
    print(
        "[info] Current curated dataset size: "
        f"{format_size_gb(curated_size_bytes)} / {args.max_curated_gb:.2f} GB"
    )

    if curated_size_bytes >= max_curated_bytes:
        print("[stop] Curated dataset already exceeds the configured size limit.")
        return

    results: list[CategoryResult] = []
    total_added = 0

    for category in categories:
        category_annotation = load_category_annotation_union(
            source_annotation_dir=args.source_annotation_dir,
            category=category,
            source_splits=source_splits,
        )
        if not category_annotation:
            print(f"[warn] No source annotations available for category {category}, skipping.")
            continue
        curated_annotation_file = args.curated_annotation_dir / f"{category}_{args.split}.jgz"
        if curated_annotation_file.exists():
            selected_entries = load_annotation_file(curated_annotation_file)
        else:
            selected_entries = {}

        curated_category_dir = args.curated_co3d_root / category
        recovered = sync_existing_annotation_entries(
            category=category,
            category_annotation=category_annotation,
            curated_category_dir=curated_category_dir,
            selected_entries=selected_entries,
            min_num_images=args.min_num_images,
        )
        if recovered:
            print(f"[info] Recovered {recovered} existing curated sequences for {category}.")

        existing_count = len(selected_entries)
        remaining_slots = max(0, args.target_sequences_per_category - existing_count)
        if remaining_slots == 0:
            results.append(
                CategoryResult(
                    category=category,
                    existing_sequences=existing_count,
                    added_sequences=0,
                    target_sequences=args.target_sequences_per_category,
                    size_limit_reached=False,
                    reason="target already satisfied",
                )
            )
            continue

        source_category_dir = args.source_co3d_root / category
        candidates = collect_available_candidates(
            category=category,
            category_annotation=category_annotation,
            source_category_dir=source_category_dir,
            selected_entries=selected_entries,
            min_num_images=args.min_num_images,
        )
        if args.shuffle:
            random.shuffle(candidates)

        if not candidates:
            results.append(
                CategoryResult(
                    category=category,
                    existing_sequences=existing_count,
                    added_sequences=0,
                    target_sequences=args.target_sequences_per_category,
                    size_limit_reached=False,
                    reason="no eligible downloaded sequences available",
                )
            )
            continue

        added_count = 0
        size_limit_reached = False
        reason = "target not yet reached"

        for seq_name, source_sequence_dir, seq_data in candidates:
            if added_count >= remaining_slots:
                reason = "target reached"
                break

            sequence_size_bytes = directory_size_bytes(source_sequence_dir)
            dest_sequence_dir = args.curated_co3d_root / category / source_sequence_dir.name
            size_delta_bytes = 0 if dest_sequence_dir.exists() else sequence_size_bytes
            projected_size = curated_size_bytes + size_delta_bytes
            if projected_size > max_curated_bytes:
                size_limit_reached = True
                reason = (
                    f"size limit reached before adding {seq_name} "
                    f"({format_size_gb(projected_size)} projected)"
                )
                break

            if dest_sequence_dir.exists():
                print(f"[warn] Destination already exists, keeping and syncing annotation: {dest_sequence_dir}")
            else:
                print(
                    f"[copy] {seq_name} -> {dest_sequence_dir} "
                    f"({format_size_gb(sequence_size_bytes)})"
                )
                if not args.dry_run:
                    copy_sequence_dir(source_sequence_dir, dest_sequence_dir)

            selected_entries[seq_name] = seq_data
            curated_size_bytes = projected_size
            added_count += 1
            total_added += 1
            print(
                f"[progress] {category}: {existing_count + added_count}/{args.target_sequences_per_category} "
                f"selected, curated size {format_size_gb(curated_size_bytes)}"
            )

        if not args.dry_run:
            write_annotation_file(curated_annotation_file, selected_entries)

        results.append(
            CategoryResult(
                category=category,
                existing_sequences=existing_count,
                added_sequences=added_count,
                target_sequences=args.target_sequences_per_category,
                size_limit_reached=size_limit_reached,
                reason=reason,
            )
        )

        if size_limit_reached:
            print(f"[stop] {reason}")
            break

    summary = {
        "source_co3d_root": str(args.source_co3d_root),
        "source_annotation_dir": str(args.source_annotation_dir),
        "curated_co3d_root": str(args.curated_co3d_root),
        "curated_annotation_dir": str(args.curated_annotation_dir),
        "split": args.split,
        "source_splits": source_splits,
        "categories": categories,
        "target_sequences_per_category": args.target_sequences_per_category,
        "min_num_images": args.min_num_images,
        "max_curated_gb": args.max_curated_gb,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "total_added_this_run": total_added,
        "final_curated_size_bytes": curated_size_bytes,
        "final_curated_size_gb": curated_size_bytes / (1024 ** 3),
        "results": [result.__dict__ for result in results],
    }

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[summary] Curation run finished.")
    print(f"[summary] Added this run: {total_added} sequences")
    print(
        "[summary] Final curated dataset size: "
        f"{format_size_gb(curated_size_bytes)} / {args.max_curated_gb:.2f} GB"
    )
    if args.report_path is not None:
        print(f"[summary] Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
