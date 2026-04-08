import json
from pathlib import Path
from typing import Dict, Iterator, Literal, Optional


def iter_caption_records(manifest_path: str | Path) -> Iterator[dict]:
    """Yield caption records from a JSONL manifest."""
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc


def load_caption_records(
    manifest_path: str | Path,
    key_field: str = "dataset_seq_name",
) -> Dict[str, dict]:
    """Load caption records into a dictionary keyed by ``key_field``."""
    records: Dict[str, dict] = {}
    for record in iter_caption_records(manifest_path):
        if key_field not in record:
            raise KeyError(
                f"Caption record is missing required key '{key_field}': {record}"
            )
        key = record[key_field]
        if key in records:
            raise ValueError(
                f"Duplicate caption manifest key '{key}' found in {manifest_path}"
            )
        records[key] = record
    return records


def load_caption_lookup(
    manifest_path: str | Path,
    key_field: str = "dataset_seq_name",
    value_field: str = "caption_concise",
) -> Dict[str, str]:
    """Load a simple ``seq_name -> caption`` lookup.

    ``value_field`` should be one of:
        ``"caption_concise"``      — short one-sentence caption (default)
        ``"caption_descriptive"``  — longer, multi-sentence caption
    """
    records = load_caption_records(manifest_path=manifest_path, key_field=key_field)
    return {
        key: record[value_field]
        for key, record in records.items()
        if value_field in record
    }


CaptionMode = Literal["concise", "descriptive"]

_CAPTION_FIELD: Dict[str, str] = {
    "concise":     "caption_concise",
    "descriptive": "caption_descriptive",
}


def load_caption_lookup_by_mode(
    manifest_path: str | Path,
    mode: CaptionMode = "concise",
    key_field: str = "dataset_seq_name",
) -> Dict[str, str]:
    """Convenience wrapper — select caption field by ``mode`` name."""
    field = _CAPTION_FIELD[mode]
    return load_caption_lookup(
        manifest_path=manifest_path,
        key_field=key_field,
        value_field=field,
    )


def get_caption_for_seq(
    caption_lookup: Dict[str, str],
    seq_name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """Return the caption associated with a dataset ``seq_name``."""
    return caption_lookup.get(seq_name, default)
