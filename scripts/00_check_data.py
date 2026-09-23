#!/usr/bin/env python
"""Step 00 — check which raw files are present / missing.

Exit codes:
  0  — ready to continue (coords may still need step 01 download)
  1  — essential non-downloadable files are missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.data.loaders import missing_coord_files, required_coord_files
from g9a_ml.paths import load_config, raw_dir_for

# Must exist locally; cannot be fetched by step 01.
REQUIRED_TABLES = [
    "data_for_one_column_target_0.csv",
    "data_for_one_column_target_1.csv",
    "pubChemComputed_target_0.csv",
    "pubChemComputed_target_1.csv",
    "pubchem_solubility.csv",
]

# Nice-to-have from the paper clone; not used by this pipeline.
OPTIONAL_TABLES = [
    "graph_feature importance.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw = raw_dir_for(cfg)
    print(f"Raw dir: {raw}")
    print(f"Exists:  {raw.exists()}")

    if not raw.exists():
        print("\nERROR: raw data directory does not exist.")
        sys.exit(1)

    missing_required: list[str] = []
    print("\nRequired tables:")
    for name in REQUIRED_TABLES:
        p = raw / name
        status = "OK" if p.exists() else "MISSING"
        size = f"{p.stat().st_size/1e6:.2f} MB" if p.exists() else "-"
        print(f"  [{status:7}] {name:40} {size}")
        if not p.exists():
            missing_required.append(name)

    print("\nOptional tables:")
    for name in OPTIONAL_TABLES:
        p = raw / name
        status = "OK" if p.exists() else "MISSING"
        size = f"{p.stat().st_size/1e6:.2f} MB" if p.exists() else "-"
        print(f"  [{status:7}] {name:40} {size}")

    print("\nCoordinate JSONs:")
    for _key, path in required_coord_files(raw, "with").items():
        status = "OK" if path.exists() else "MISSING"
        size = f"{path.stat().st_size/1e6:.2f} MB" if path.exists() else "-"
        print(f"  [{status:7}] {path.name:40} {size}")

    if missing_required:
        print("\nERROR: required tables are missing. Copy them from G9a_clsf/with solubility/")
        for name in missing_required:
            print(f"  - {name}")
        sys.exit(1)

    missing_coords = missing_coord_files(raw, "with")
    if missing_coords:
        print("\nMissing coordinate files (expected — step 01 will download them):")
        for path in missing_coords:
            print(f"  - {path.name}")
        print("Next: python scripts/01_download_coordinates.py")
        # Exit 0 so run_all.py can continue into the download step.
        sys.exit(0)

    print("\nAll required files present. Next: python scripts/02_build_features.py")
    print("(Or shortcut: python scripts/create_dataset.py)")


if __name__ == "__main__":
    main()
