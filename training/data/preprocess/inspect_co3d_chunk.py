"""
Inspect a downloaded Co3D chunk and compare it against VGGT-style annotation files.

This is a debugging utility for answering questions like:
- How many sequence folders are actually present in this chunk?
- How many frames does each sequence contain?
- How many chunk sequences overlap with the external ``co3d_anno`` files?
- How many overlapping sequences satisfy the VGGT minimum image threshold?

Example:
    python training/data/preprocess/inspect_co3d_chunk.py \
        --chunk-root /tmp/apple_000/extracted \
        --category apple \
        --annotation-file /path/to/co3d_anno/apple_train.jgz \
        --min-num-images 24
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-root", type=Path, required=True, help="Extracted chunk root.")
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="Optional external <category>_<split>.jgz file for overlap analysis.",
    )
    parser.add_argument(
        "--annotation-files",
        nargs="*",
        default=None,
        help="Optional list of annotation files to union for overlap analysis.",
    )
    parser.add_argument("--min-num-images", type=int, default=24)
    parser.add_argument(
        "--show-samples",
        type=int,
        default=10,
        help="How many sample sequence names to print.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def load_annotation_file(annotation_file: Path) -> dict:
    with gzip.open(annotation_file, "rt", encoding="utf-8") as handle:
        return json.loads(handle.read())


def load_annotation_union(annotation_files: list[Path]) -> dict:
    merged = {}
    for annotation_file in annotation_files:
        payload = load_annotation_file(annotation_file)
        merged.update(payload)
    return merged


def count_images(images_dir: Path) -> int:
    return sum(
        1
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def inspect_chunk_sequences(chunk_root: Path, category: str) -> list[dict]:
    category_dir = chunk_root / category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Category directory not found inside chunk: {category_dir}")

    sequence_infos: list[dict] = []
    for sequence_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
        images_dir = sequence_dir / "images"
        if not images_dir.is_dir():
            continue
        image_count = count_images(images_dir)
        sequence_infos.append(
            {
                "sequence_dir_name": sequence_dir.name,
                "image_count": image_count,
            }
        )
    return sequence_infos


def main() -> None:
    args = parse_args()
    sequence_infos = inspect_chunk_sequences(args.chunk_root.resolve(), args.category)

    summary = {
        "chunk_root": str(args.chunk_root.resolve()),
        "category": args.category,
        "sequence_dir_count": len(sequence_infos),
        "eligible_sequence_dir_count": sum(
            info["image_count"] >= args.min_num_images for info in sequence_infos
        ),
        "min_num_images": args.min_num_images,
        "sample_sequence_dirs": [
            info["sequence_dir_name"] for info in sequence_infos[: args.show_samples]
        ],
        "sample_image_counts": {
            info["sequence_dir_name"]: info["image_count"]
            for info in sequence_infos[: args.show_samples]
        },
    }

    annotation_files: list[Path] = []
    if args.annotation_files:
        annotation_files.extend(args.annotation_files)
    if args.annotation_file is not None:
        annotation_files.append(args.annotation_file)

    if annotation_files:
        annotation = load_annotation_union(annotation_files)
        annotation_keys = set(annotation.keys())
        chunk_sequence_names = [info["sequence_dir_name"] for info in sequence_infos]

        overlap_keys = [seq_name for seq_name in chunk_sequence_names if seq_name in annotation_keys]
        overlap_eligible = [
            seq_name
            for seq_name in overlap_keys
            if len(annotation[seq_name]) >= args.min_num_images
        ]

        summary.update(
            {
                "annotation_files": [str(path.resolve()) for path in annotation_files],
                "annotation_sequence_count": len(annotation),
                "overlap_sequence_count": len(overlap_keys),
                "overlap_eligible_count": len(overlap_eligible),
                "sample_overlap_sequence_names": overlap_keys[: args.show_samples],
                "sample_overlap_image_counts": {
                    seq_name: next(
                        info["image_count"]
                        for info in sequence_infos
                        if info["sequence_dir_name"] == seq_name
                    )
                    for seq_name in overlap_keys[: args.show_samples]
                },
            }
        )

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
