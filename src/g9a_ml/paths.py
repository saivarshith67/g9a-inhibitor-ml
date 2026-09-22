"""Path helpers and config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    root = project_root()
    path = Path(config_path) if config_path else root / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_root"] = root
    for key in (
        "raw_dir",
        "interim_dir",
        "processed_dir",
        "metrics_dir",
        "models_dir",
        "figures_dir",
    ):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else root / p
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in (
        "raw_dir",
        "interim_dir",
        "processed_dir",
        "metrics_dir",
        "models_dir",
        "figures_dir",
    ):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)


def raw_dir_for(cfg: dict[str, Any], solubility: str | None = None) -> Path:
    """Resolve raw data folder for with/without solubility."""
    sol = solubility or cfg["dataset"]["solubility"]
    root = cfg["_root"]
    if sol == "with":
        # Prefer config raw_dir; fall back to data/raw layout
        configured = cfg.get("raw_dir")
        if configured and Path(configured).exists():
            return Path(configured)
        return root / "data" / "raw" / "with_solubility"
    if sol == "without":
        return root / "data" / "raw" / "without_solubility"
    raise ValueError(f"Unknown solubility setting: {sol}")


def dataset_stem(cfg: dict[str, Any]) -> str:
    sol = cfg["dataset"]["solubility"]
    bal = cfg["dataset"]["balancing"]
    return f"{sol}_solubility_{bal}"
