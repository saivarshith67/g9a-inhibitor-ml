"""Coordinate-derived features from PubChem 2D substance / 3D compound JSON."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import skew


def features_from_2d_json(payload: dict[str, Any]) -> pd.DataFrame:
    """Extract MMX/MMY and skew features from a PC_Substances JSON payload."""
    rows: list[list[float]] = []
    for substance in payload.get("PC_Substances", []):
        sid = substance["sid"]["id"]
        compound0 = substance["compound"][0]
        element = compound0["atoms"]["element"]

        if "coords" not in compound0:
            continue

        conformers_x = compound0["coords"][0]["conformers"][0]["x"]
        conformers_y = compound0["coords"][0]["conformers"][0]["y"]

        sixes_x: list[float] = []
        sixes_y: list[float] = []
        seq = False
        for i, el in enumerate(element):
            if el == 6:
                seq = True
                sixes_x.append(conformers_x[i])
                sixes_y.append(conformers_y[i])
            elif seq:
                break

        if not sixes_x:
            continue

        rows.append(
            [
                sid,
                max(sixes_x) - min(sixes_x),
                max(conformers_x) - min(conformers_x),
                float(skew(sixes_x)),
                float(skew(conformers_x)),
                max(sixes_y) - min(sixes_y),
                max(conformers_y) - min(conformers_y),
                float(skew(sixes_y)),
                float(skew(conformers_y)),
            ]
        )

    return pd.DataFrame(
        rows,
        columns=["SID", "MMX6", "MMX", "SX6", "SX", "MMY6", "MMY", "SY6", "SY"],
    )


def features_from_3d_json(payload: dict[str, Any]) -> pd.DataFrame:
    """Extract 3D coordinate span/skew features from a PC_Compounds JSON payload."""
    rows: list[list[float]] = []
    for compound in payload.get("PC_Compounds", []):
        cid = compound["id"]["id"]["cid"]
        element = compound["atoms"]["element"]
        conf = compound["coords"][0]["conformers"][0]
        conformers_x, conformers_y, conformers_z = conf["x"], conf["y"], conf["z"]

        sixes_x: list[float] = []
        sixes_y: list[float] = []
        sixes_z: list[float] = []
        seq = False
        for i, el in enumerate(element):
            if el == 6:
                seq = True
                sixes_x.append(conformers_x[i])
                sixes_y.append(conformers_y[i])
                sixes_z.append(conformers_z[i])
            elif seq:
                break

        if not sixes_x:
            continue

        rows.append(
            [
                cid,
                max(sixes_x) - min(sixes_x),
                max(conformers_x) - min(conformers_x),
                float(skew(sixes_x)),
                float(skew(conformers_x)),
                max(sixes_y) - min(sixes_y),
                max(conformers_y) - min(conformers_y),
                float(skew(sixes_y)),
                float(skew(conformers_y)),
                max(sixes_z) - min(sixes_z),
                max(conformers_z) - min(conformers_z),
                float(skew(sixes_z)),
                float(skew(conformers_z)),
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "CID",
            "MMX6_3D",
            "MMX_3D",
            "SX6_3D",
            "SX_3D",
            "MMY6_3D",
            "MMY_3D",
            "SY6_3D",
            "SY_3D",
            "MMZ6_3D",
            "MMZ_3D",
            "SZ6_3D",
            "SZ_3D",
        ],
    )
