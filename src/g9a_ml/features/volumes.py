"""Hypothetical volume and size-ratio features (paper formulas preserved)."""

from __future__ import annotations

import math

import pandas as pd

PI = math.pi


def add_2d_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """Add Volume_1 and Volume_2 from 2D coordinate features.

    Note: the paper uses ``x**1/2`` (i.e. x/2), not ``sqrt(x)``. Kept as-is.
    """
    out = df.copy()
    along_x = (out["MMX"] - out["MMX6"]) ** 2
    along_y = (out["MMY"] - out["MMY6"]) ** 2
    diagonal = (along_x + along_y) ** 1 / 2
    radius = diagonal / 2
    out["Volume_1"] = (4 * PI * radius**3) / 3

    dist_x = (
        out["MMX6"] / 2
        - out["MMX"] / 2
        - out["SX6"]
        + out["SX"]
    )
    dist_y = (
        out["MMY6"] / 2
        - out["MMY"] / 2
        - out["SY6"]
        + out["SY"]
    )
    xy_2d = ((dist_x**2 + dist_y**2) ** 1 / 2) * 1 / 2
    out["Volume_2"] = (PI * (xy_2d) ** 3) * 1 / 3
    return out


def add_3d_volumes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    along_x = (out["MMX_3D"] - out["MMX6_3D"]) ** 2
    along_y = (out["MMY_3D"] - out["MMY6_3D"]) ** 2
    along_z = (out["MMZ_3D"] - out["MMZ6_3D"]) ** 2
    out["Volume_1_3D"] = along_x * along_y * along_z

    dist_x = out["MMX6_3D"] / 2 - out["MMX_3D"] / 2 - out["SX6_3D"] + out["SX_3D"]
    dist_y = out["MMY6_3D"] / 2 - out["MMY_3D"] / 2 - out["SY6_3D"] + out["SY_3D"]
    dist_z = out["MMZ6_3D"] / 2 - out["MMZ_3D"] / 2 - out["SZ6_3D"] + out["SZ_3D"]

    xy_3d = ((dist_x**2 + dist_y**2) ** 1 / 2) * 1 / 2
    xz_3d = ((dist_x**2 + dist_z**2) ** 1 / 2) * 1 / 2
    yz_3d = ((dist_z**2 + dist_y**2) ** 1 / 2) * 1 / 2

    out["XY_3D_volume"] = (PI * (xy_3d) ** 3) * 1 / 3
    out["XZ_3D_volume"] = (PI * (xz_3d) ** 3) * 1 / 3
    out["YZ_3D_volume"] = (PI * (yz_3d) ** 3) * 1 / 3
    return out


def add_size_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["C_rel_2D"] = out["MMX6"] / out["MMY6"]
    out["allAtoms_rel_2D"] = out["MMX"] / out["MMY"]
    out["C_rel_XY_3D"] = out["MMX6_3D"] / out["MMY6_3D"]
    out["allAtoms_rel_XY_3D"] = out["MMX_3D"] / out["MMY_3D"]
    out["C_rel_XZ_3D"] = out["MMX6_3D"] / out["MMZ6_3D"]
    out["allAtoms_rel_XZ_3D"] = out["MMX_3D"] / out["MMZ_3D"]
    out["C_rel_YZ_3D"] = out["MMY6_3D"] / out["MMZ6_3D"]
    out["allAtoms_rel_YZ_3D"] = out["MMY_3D"] / out["MMZ_3D"]
    return out
