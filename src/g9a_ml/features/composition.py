"""Atom relative counts and mass fractions from molecular formulas."""

from __future__ import annotations

import re

import pandas as pd
from chemformula import ChemFormula

ELEMENTS = ["C", "H", "O", "S", "N", "Br", "Cl", "F"]


def _parse_formula(formula: str) -> dict[str, int]:
    parts = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return {sym: int(n) if n else 1 for sym, n in parts}


def _atom_total(counts: dict[str, int]) -> int:
    return sum(counts.values())


def relative_atom_proportions(formulas: list[str]) -> pd.DataFrame:
    """Relative atom counts (replaces notebook ``chem_calc`` dependency)."""
    rows = []
    for formula in formulas:
        counts = _parse_formula(formula)
        total = _atom_total(counts) or 1
        row = {"MF": formula}
        for el in ELEMENTS:
            row[f"{el}_relative"] = round(counts.get(el, 0) / total, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def mass_proportions(formulas: list[str]) -> pd.DataFrame:
    rows = []
    for formula in formulas:
        cf = ChemFormula(formula, name=formula)
        fractions = {el: 0.0 for el in ELEMENTS}
        for symbol, frac in cf.mass_fraction.items():
            if symbol in fractions:
                fractions[symbol] = float(round(frac * 100, 2))
        row = {"MF": formula, **fractions}
        rows.append(row)
    return pd.DataFrame(rows)
