#!/usr/bin/env python
"""Step 00 — check which raw files are present / missing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g9a_ml.data.loaders import missing_coord_files, required_coord_files
from g9a_ml.paths import load_config, raw_dir_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw = raw_dir_for(cfg)
    print(f"Raw dir: {raw}")
    print(f"Exists:  {raw.exists()}")

    expected = [
        "data_for_one_column_target_0.csv",
        "data_for_one_column_target_1.csv",
        "pubChemComputed_target_0.csv",
        "pubChemComputed_target_1.csv",
        "pubchem_solubility.csv",
        "graph_feature importance.csv",
    ]
    for name in expected:
        p = raw / name
        status = "OK" if p.exists() else "MISSING"
        size = f"{p.stat().st_size/1e6:.2f} MB" if p.exists() else "-"
        print(f"  [{status:7}] {name:40} {size}")

    print("\nCoordinate JSONs:")
    for key, path in required_coord_files(raw, "with").items():
        status = "OK" if path.exists() else "MISSING"
        size = f"{path.stat().st_size/1e6:.2f} MB" if path.exists() else "-"
        print(f"  [{status:7}] {path.name:40} {size}")

    missing = missing_coord_files(raw, "with")
    if missing:
        print("\nNext: python scripts/01_download_coordinates.py")
        sys.exit(1)
    print("\nAll required files present. Next: python scripts/02_build_features.py")
    print("(Or shortcut: python scripts/create_dataset.py)")


if __name__ == "__main__":
    main()
