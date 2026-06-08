"""YAML configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_FILES = ("base.yaml", "data.yaml", "model.yaml")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {yaml_path}")
    return data


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""

    merged = deepcopy(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config_set(
    root: str | Path,
    config_files: tuple[str, ...] = DEFAULT_CONFIG_FILES,
) -> dict[str, Any]:
    """Load and merge the standard config files from `configs/`."""

    project_root = Path(root).resolve()
    config_dir = project_root / "configs"
    merged: dict[str, Any] = {}
    for file_name in config_files:
        merged = deep_merge(merged, load_yaml(config_dir / file_name))
    return merged


def require_config_keys(config: dict[str, Any], keys: list[str]) -> None:
    """Raise a clear error if required top-level config keys are missing."""

    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")
