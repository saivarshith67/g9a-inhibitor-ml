"""Assemble final coordinate JSON files from per-batch download caches."""

from __future__ import annotations

import json
from pathlib import Path


def assemble_json_from_cache(
    cache_dir: Path,
    out_path: Path,
    wrapper_key: str,
) -> Path | None:
    """Merge ``batch_XXXXX.json`` files into a single PC_Substances / PC_Compounds JSON.

    Returns the output path if assembly ran (or file already exists), else None
    if the cache directory is missing/empty.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    if not cache_dir.is_dir():
        return None

    batches = sorted(cache_dir.glob("batch_*.json"))
    if not batches:
        return None

    records: list = []
    for batch_file in batches:
        with open(batch_file, encoding="utf-8") as f:
            records.extend(json.load(f))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({wrapper_key: records}, f)
    return out_path


def ensure_coordinate_jsons(raw_dir: Path) -> dict[str, Path]:
    """Ensure SID/CID target_0 JSON files exist, assembling from cache if needed."""
    pairs = [
        (
            raw_dir / ".cache_SID_2D_target_0",
            raw_dir / "SID_2D_target_0.json",
            "PC_Substances",
        ),
        (
            raw_dir / ".cache_CID_3D_target_0",
            raw_dir / "CID_3D_target_0.json",
            "PC_Compounds",
        ),
    ]
    result: dict[str, Path] = {}
    for cache_dir, out_path, key in pairs:
        path = assemble_json_from_cache(cache_dir, out_path, key)
        if path is None or not out_path.exists():
            raise FileNotFoundError(
                f"Missing {out_path.name} and no usable cache at {cache_dir}. "
                "Run: python scripts/01_download_coordinates.py"
            )
        result[out_path.name] = out_path
    return result
