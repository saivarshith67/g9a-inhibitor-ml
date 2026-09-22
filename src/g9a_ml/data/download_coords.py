"""Download missing PubChem 2D substance / 3D compound coordinate JSON files."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _get_json(url: str, max_retries: int = 6) -> dict | None:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=180)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (404, 400):
                return None
            # rate limit / transient
            time.sleep(1.0 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None


def _fetch_and_cache_batch(
    bi: int,
    ids: list[int],
    cache_file: Path,
    kind: str,
) -> tuple[int, list]:
    """Download one batch and write cache file. Returns (batch_index, records)."""
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            return bi, json.load(f)

    id_str = ",".join(str(i) for i in ids)
    if kind == "sid":
        url = f"{PUBCHEM}/substance/sid/{id_str}/JSON"
        key = "PC_Substances"
    else:
        url = f"{PUBCHEM}/compound/cid/{id_str}/JSON?record_type=3d"
        key = "PC_Compounds"

    payload = _get_json(url)
    batch_items = payload.get(key, []) if payload else []
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(batch_items, f)
    return bi, batch_items


def _download_batched(
    ids: list[int],
    out_path: Path,
    kind: str,
    batch_size: int = 50,
    workers: int = 8,
    desc: str = "",
) -> Path:
    """Parallel batched download with per-batch disk cache (resumable)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out_path.parent / f".cache_{out_path.stem}"
    cache_dir.mkdir(exist_ok=True)

    batches = list(_chunks(ids, batch_size))
    results: dict[int, list] = {}

    # Load already-cached batches first
    pending: list[tuple[int, list[int], Path]] = []
    for bi, batch in enumerate(batches):
        cache_file = cache_dir / f"batch_{bi:05d}.json"
        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                results[bi] = json.load(f)
        else:
            pending.append((bi, batch, cache_file))

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_and_cache_batch, bi, batch, cache_file, kind): bi
                for bi, batch, cache_file in pending
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc or out_path.name,
            ):
                bi, items = fut.result()
                results[bi] = items

    # Assemble in order
    records: list = []
    for bi in range(len(batches)):
        records.extend(results.get(bi, []))

    wrapper_key = "PC_Substances" if kind == "sid" else "PC_Compounds"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({wrapper_key: records}, f)
    return out_path


def download_substance_2d(
    sids: list[int],
    out_path: Path,
    batch_size: int = 50,
    sleep_seconds: float = 0.0,
    workers: int = 8,
) -> Path:
    """Download substance records (includes 2D coords) and write PC_Substances JSON."""
    _ = sleep_seconds  # kept for API compatibility; parallel path uses retries
    return _download_batched(
        sids,
        out_path,
        kind="sid",
        batch_size=batch_size,
        workers=workers,
        desc=f"SID 2D -> {out_path.name}",
    )


def download_compound_3d(
    cids: list[int],
    out_path: Path,
    batch_size: int = 50,
    sleep_seconds: float = 0.0,
    workers: int = 8,
) -> Path:
    """Download 3D compound records and write PC_Compounds JSON."""
    _ = sleep_seconds
    return _download_batched(
        cids,
        out_path,
        kind="cid",
        batch_size=batch_size,
        workers=workers,
        desc=f"CID 3D -> {out_path.name}",
    )


def download_missing_with_solubility_coords(
    raw_dir: Path,
    batch_size: int = 50,
    sleep_seconds: float = 0.0,
    limit: int | None = None,
    workers: int = 8,
) -> list[Path]:
    """Fetch SID_2D_target_0.json and CID_3D_target_0.json if missing.

    ``limit`` restricts IDs for a quick smoke-test download.
    """
    written: list[Path] = []
    t0 = pd.read_csv(raw_dir / "data_for_one_column_target_0.csv")
    sids = t0["SID"].astype(int).tolist()
    cids = t0["CID"].astype(int).tolist()
    if limit is not None:
        sids, cids = sids[:limit], cids[:limit]

    sid_path = raw_dir / "SID_2D_target_0.json"
    cid_path = raw_dir / "CID_3D_target_0.json"

    if not sid_path.exists():
        written.append(
            download_substance_2d(
                sids,
                sid_path,
                batch_size=batch_size,
                sleep_seconds=sleep_seconds,
                workers=workers,
            )
        )
    if not cid_path.exists():
        written.append(
            download_compound_3d(
                cids,
                cid_path,
                batch_size=batch_size,
                sleep_seconds=sleep_seconds,
                workers=workers,
            )
        )

    meta = {
        "n_target0_requested": len(sids),
        "limit": limit,
        "complete": limit is None,
        "sid_2d_exists": sid_path.exists(),
        "cid_3d_exists": cid_path.exists(),
        "sid_2d_bytes": sid_path.stat().st_size if sid_path.exists() else 0,
        "cid_3d_bytes": cid_path.stat().st_size if cid_path.exists() else 0,
    }
    with open(raw_dir / "target0_coords_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return written
