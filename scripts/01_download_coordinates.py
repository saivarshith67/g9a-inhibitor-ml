#!/usr/bin/env python
"""Step 01 — download missing PubChem coordinate JSON files (target_0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.data.download_coords import download_missing_with_solubility_coords
from g9a_ml.data.loaders import missing_coord_files
from g9a_ml.paths import ensure_dirs, load_config, raw_dir_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N target_0 IDs (smoke test). Overwrites nothing if files exist.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing target_0 coordinate files and re-download.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = raw_dir_for(cfg, "with")

    sid_path = raw / "SID_2D_target_0.json"
    cid_path = raw / "CID_3D_target_0.json"
    if args.force:
        for p in (sid_path, cid_path):
            if p.exists():
                p.unlink()
                print(f"Removed {p}")

    missing = missing_coord_files(raw, "with")
    if not missing and not args.force:
        print("All coordinate files present:")
        for p in sorted(raw.glob("*.json")):
            print(f"  {p.name}")
        return

    print(f"Downloading missing coordinates into: {raw}")
    if args.limit:
        print(f"LIMIT={args.limit} (smoke-test mode; not full paper dataset)")
    written = download_missing_with_solubility_coords(
        raw,
        batch_size=cfg["download"]["batch_size"],
        sleep_seconds=cfg["download"].get("sleep_seconds", 0.0),
        limit=args.limit,
        workers=cfg["download"].get("workers", 8),
    )
    if written:
        for p in written:
            print(f"Wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
    else:
        print("Nothing to download.")

    still = missing_coord_files(raw, "with")
    if still:
        print("Still missing:", ", ".join(p.name for p in still))
        sys.exit(1)
    print("Coordinate files ready.")


if __name__ == "__main__":
    main()
