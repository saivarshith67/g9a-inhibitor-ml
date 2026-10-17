"""Load project ``.env`` (e.g. CUDA_VISIBLE_DEVICES) before PyTorch/CUDA init."""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def load_project_env() -> None:
    """Load ``<project_root>/.env`` once; existing shell env vars take precedence."""
    global _loaded
    if _loaded:
        return
    env_file = project_root() / ".env"
    if env_file.is_file():
        _load_env_file(env_file)
    _loaded = True
