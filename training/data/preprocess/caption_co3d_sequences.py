"""
Offline sequence captioning for Co3D data used by VGGT-style training.

This script supports two input modes:
1. The VGGT training setup with external ``*.jgz`` annotation files.
2. A raw Co3D directory scan, which is useful for the official single-sequence subset.

In either case it:
1. Enumerates valid sequences.
2. Samples a few representative frames from each sequence.
3. Runs a local Qwen2.5-VL model to produce one descriptive caption per sequence.
4. Writes a JSONL manifest keyed by ``dataset_seq_name``.

The intent is to run this on a GPU machine such as SCC, not inside the training loop.

Examples:
    python training/data/preprocess/caption_co3d_sequences.py \
        --co3d-dir /path/to/co3d \
        --co3d-annotation-dir /path/to/co3d_anno \
        --output-path /path/to/co3d_train_captions.jsonl \
        --split train \
        --frames-per-sequence 4 \
        --max-sequences 1000 \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct

    python training/data/preprocess/caption_co3d_sequences.py \
        --raw-co3d-root /path/to/co3d_subset \
        --output-path /path/to/co3d_subset_captions.jsonl \
        --frames-per-sequence 4 \
        --max-sequences 50
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


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


DEFAULT_PROMPT_TEMPLATE = (
    "These images are different views from the same 3D capture sequence. "
    "Write one concise caption describing the shared main object and stable visible context across the views. "
    "Mention the object type and clear visual attributes such as color, material, shape, or scene context only when they are clearly supported by the images. "
    "Do not mention camera motion, viewpoint changes, or uncertain details. "
    "Output exactly one sentence."
)


@dataclass
class SequenceRecord:
    category: str
    split: str
    source_seq_name: str
    dataset_seq_name: str
    image_paths: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--co3d-dir",
        type=Path,
        default=None,
        help="Root directory of Co3D images. Required when using --co3d-annotation-dir.",
    )
    parser.add_argument(
        "--co3d-annotation-dir",
        type=Path,
        default=None,
        help="Directory containing <category>_<split>.jgz annotation files used by VGGT.",
    )
    parser.add_argument(
        "--raw-co3d-root",
        type=Path,
        default=None,
        help="Raw Co3D root to scan directly, useful for the official single-sequence subset.",
    )
    parser.add_argument("--output-path", type=Path, required=True, help="Output JSONL manifest path.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category subset. Defaults to the VGGT seen categories.",
    )
    parser.add_argument(
        "--min-num-images",
        type=int,
        default=24,
        help="Skip sequences shorter than this, matching VGGT defaults.",
    )
    parser.add_argument(
        "--frames-per-sequence",
        type=int,
        default=4,
        help="Number of representative frames to show the VLM per sequence.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional total sequence cap for pilot runs.",
    )
    parser.add_argument(
        "--max-sequences-per-category",
        type=int,
        default=None,
        help="Optional per-category cap for more balanced pilot runs.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle sequence order before applying any max-sequence limits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Local Hugging Face model id to use for captioning.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Transformers device_map value. Use 'auto' for SCC GPU inference.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attention backend such as flash_attention_2.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Generation cap for the caption output.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional Qwen processor min_pixels override.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional Qwen processor max_pixels override.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file overriding the default prompt template.",
    )
    parser.add_argument(
        "--use-category-hint",
        action="store_true",
        help="Append the known Co3D category label to the prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing manifest instead of resuming.",
    )
    args = parser.parse_args()

    if args.co3d_annotation_dir is not None and args.co3d_dir is None:
        parser.error("--co3d-dir is required when --co3d-annotation-dir is provided")
    if args.co3d_annotation_dir is None and args.raw_co3d_root is None:
        parser.error("Provide either --co3d-annotation-dir with --co3d-dir, or --raw-co3d-root")
    if args.co3d_annotation_dir is not None and args.raw_co3d_root is not None:
        parser.error("Use either annotation mode or raw-scan mode, not both")

    return args


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is None:
        return DEFAULT_PROMPT_TEMPLATE
    return args.prompt_file.read_text(encoding="utf-8").strip()


def load_annotation_file(annotation_file: Path) -> Dict[str, list]:
    with gzip.open(annotation_file, "rt", encoding="utf-8") as handle:
        return json.loads(handle.read())


def collect_sequences_from_annotations(args: argparse.Namespace) -> list[SequenceRecord]:
    categories = sorted(args.categories or DEFAULT_SEEN_CATEGORIES)
    all_sequences: list[SequenceRecord] = []

    for category in categories:
        annotation_file = args.co3d_annotation_dir / f"{category}_{args.split}.jgz"
        if not annotation_file.exists():
            print(f"[warn] Annotation file not found, skipping: {annotation_file}")
            continue

        annotation = load_annotation_file(annotation_file)
        category_records = []
        for seq_name, seq_data in annotation.items():
            if len(seq_data) < args.min_num_images:
                continue
            image_paths = [str((args.co3d_dir / anno["filepath"]).resolve()) for anno in seq_data]
            category_records.append(
                SequenceRecord(
                    category=category,
                    split=args.split,
                    source_seq_name=seq_name,
                    dataset_seq_name=f"co3d_{seq_name}",
                    image_paths=image_paths,
                )
            )

        if args.shuffle:
            random.shuffle(category_records)

        if args.max_sequences_per_category is not None:
            category_records = category_records[: args.max_sequences_per_category]

        all_sequences.extend(category_records)

    if args.shuffle:
        random.shuffle(all_sequences)

    if args.max_sequences is not None:
        all_sequences = all_sequences[: args.max_sequences]

    return all_sequences


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_sequences_from_raw_root(args: argparse.Namespace) -> list[SequenceRecord]:
    root = args.raw_co3d_root.resolve()
    categories = sorted(args.categories) if args.categories else sorted(
        path.name for path in root.iterdir() if path.is_dir()
    )
    all_sequences: list[SequenceRecord] = []

    for category in categories:
        category_dir = root / category
        if not category_dir.is_dir():
            continue

        category_records = []
        for sequence_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            images_dir = sequence_dir / "images"
            if not images_dir.is_dir():
                continue
            image_paths = sorted(str(path.resolve()) for path in images_dir.iterdir() if path.is_file() and is_image_file(path))
            if len(image_paths) < args.min_num_images:
                continue

            source_seq_name = f"{category}/{sequence_dir.name}"
            category_records.append(
                SequenceRecord(
                    category=category,
                    split="subset",
                    source_seq_name=source_seq_name,
                    dataset_seq_name=f"co3d_{source_seq_name}",
                    image_paths=image_paths,
                )
            )

        if args.shuffle:
            random.shuffle(category_records)

        if args.max_sequences_per_category is not None:
            category_records = category_records[: args.max_sequences_per_category]

        all_sequences.extend(category_records)

    if args.shuffle:
        random.shuffle(all_sequences)

    if args.max_sequences is not None:
        all_sequences = all_sequences[: args.max_sequences]

    return all_sequences


def collect_sequences(args: argparse.Namespace) -> list[SequenceRecord]:
    if args.co3d_annotation_dir is not None:
        return collect_sequences_from_annotations(args)
    return collect_sequences_from_raw_root(args)


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = min(length, max(1, count))
    raw = np.linspace(0, length - 1, num=count, dtype=int).tolist()
    deduped = []
    seen = set()
    for idx in raw:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def build_prompt(prompt_template: str, category: str, use_category_hint: bool) -> str:
    if not use_category_hint:
        return prompt_template
    return f"{prompt_template} The known category label is '{category}'."


def read_existing_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "dataset_seq_name" in record:
                completed.add(record["dataset_seq_name"])
    return completed


def load_qwen_components(args: argparse.Namespace):
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "This script requires Qwen VLM dependencies. Install them with:\n"
            "  pip install git+https://github.com/huggingface/transformers accelerate\n"
            "  pip install qwen-vl-utils"
        ) from exc

    model_kwargs = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_name, **processor_kwargs)

    return model, processor, process_vision_info


def make_message(image_paths: Sequence[Path], prompt_text: str) -> list[dict]:
    content = [{"type": "image", "image": image_path.resolve().as_uri()} for image_path in image_paths]
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def generate_caption(
    model,
    processor,
    process_vision_info,
    image_paths: Sequence[Path],
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    messages = make_message(image_paths=image_paths, prompt_text=prompt_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def write_metadata(
    args: argparse.Namespace,
    output_path: Path,
    sequences: Sequence[SequenceRecord],
    prompt_text: str,
) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    selected_categories = sorted({record.category for record in sequences})
    metadata = {
        "script": "caption_co3d_sequences.py",
        "model_name": args.model_name,
        "split": args.split if args.co3d_annotation_dir is not None else "subset",
        "input_mode": "annotations" if args.co3d_annotation_dir is not None else "raw_scan",
        "co3d_dir": str(args.co3d_dir) if args.co3d_dir is not None else None,
        "co3d_annotation_dir": str(args.co3d_annotation_dir) if args.co3d_annotation_dir is not None else None,
        "raw_co3d_root": str(args.raw_co3d_root) if args.raw_co3d_root is not None else None,
        "categories": selected_categories,
        "min_num_images": args.min_num_images,
        "frames_per_sequence": args.frames_per_sequence,
        "max_sequences": args.max_sequences,
        "max_sequences_per_category": args.max_sequences_per_category,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "use_category_hint": args.use_category_hint,
        "prompt": prompt_text,
        "total_sequences_selected": len(sequences),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = load_prompt(args)
    sequences = collect_sequences(args)
    write_metadata(args, output_path, sequences, prompt_template)

    if not sequences:
        print("[info] No sequences selected. Nothing to caption.")
        return

    completed_keys = set()
    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        else:
            completed_keys = read_existing_keys(output_path)

    model, processor, process_vision_info = load_qwen_components(args)

    mode = "a" if output_path.exists() and not args.overwrite else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        num_done = len(completed_keys)
        total = len(sequences)

        for index, record in enumerate(sequences, start=1):
            if record.dataset_seq_name in completed_keys:
                if index % 100 == 0:
                    print(f"[resume] {index}/{total} scanned, {num_done} already completed")
                continue

            sampled_indices = evenly_spaced_indices(len(record.image_paths), args.frames_per_sequence)
            sampled_image_paths = [
                Path(record.image_paths[idx]).resolve()
                for idx in sampled_indices
            ]

            missing_paths = [str(path) for path in sampled_image_paths if not path.exists()]
            if missing_paths:
                print(f"[warn] Missing images for {record.dataset_seq_name}, skipping")
                output_record = {
                    "dataset": "co3d",
                    "split": record.split,
                    "category": record.category,
                    "source_seq_name": record.source_seq_name,
                    "dataset_seq_name": record.dataset_seq_name,
                    "num_frames_in_sequence": len(record.image_paths),
                    "sampled_frame_indices": sampled_indices,
                    "sampled_image_paths": [str(path) for path in sampled_image_paths],
                    "caption": None,
                    "status": "missing_images",
                    "missing_image_paths": missing_paths,
                    "model_name": args.model_name,
                }
                handle.write(json.dumps(output_record, ensure_ascii=True) + "\n")
                handle.flush()
                continue

            prompt_text = build_prompt(
                prompt_template=prompt_template,
                category=record.category,
                use_category_hint=args.use_category_hint,
            )

            try:
                caption = generate_caption(
                    model=model,
                    processor=processor,
                    process_vision_info=process_vision_info,
                    image_paths=sampled_image_paths,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                )
                status = "ok"
                error_message = None
            except Exception as exc:  # pragma: no cover - runtime failure path
                caption = None
                status = "error"
                error_message = str(exc)

            output_record = {
                "dataset": "co3d",
                "split": record.split,
                "category": record.category,
                "source_seq_name": record.source_seq_name,
                "dataset_seq_name": record.dataset_seq_name,
                "num_frames_in_sequence": len(record.image_paths),
                "sampled_frame_indices": sampled_indices,
                "sampled_image_paths": [str(path) for path in sampled_image_paths],
                "caption": caption,
                "status": status,
                "error": error_message,
                "model_name": args.model_name,
            }
            handle.write(json.dumps(output_record, ensure_ascii=True) + "\n")
            handle.flush()

            if status == "ok":
                num_done += 1

            if index == 1 or index % 25 == 0:
                print(
                    f"[progress] processed {index}/{total} sequences, "
                    f"successful captions so far: {num_done}"
                )


if __name__ == "__main__":
    main()
