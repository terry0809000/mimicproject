from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_all_configs(config_dir: str | Path = "config") -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for name in ["base", "data", "models", "evaluation", "capability_tests", "logging"]:
        cfg = merge_dicts(cfg, load_yaml(Path(config_dir) / f"{name}.yaml"))
    return cfg
