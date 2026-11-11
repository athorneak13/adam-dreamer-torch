"""Load a YAML config into nested attribute-access namespaces."""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path) as f:
        return _to_namespace(yaml.safe_load(f))


def apply_overrides(cfg: SimpleNamespace, overrides: list[str]) -> SimpleNamespace:
    """Apply 'a.b.c=value' strings to cfg in place. Values are parsed as YAML
    (so 5 -> int, 3e-4 -> str, 3.0e-4 -> float, true -> bool). Unknown keys raise."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like key=value")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for name in parents:
            if not hasattr(node, name):
                raise KeyError(f"unknown config key '{key}'")
            node = getattr(node, name)
        if not hasattr(node, leaf):
            raise KeyError(f"unknown config key '{key}'")
        setattr(node, leaf, _to_namespace(yaml.safe_load(raw)))
    return cfg
