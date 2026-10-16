"""Load, combine, override, and save the YAML settings of a run.

A configuration is a plain nested dict. It comes from:

1. one or more YAML files in `configs/` (a later file overrides an earlier one);
2. command-line overrides of the form `section.key=value`.

The text of an override value is read as YAML, so `train.epochs=30` gives an int,
`data.datasets=[pix3d,abo]` gives a list, and `render.lighting=L2` gives a string.
PyYAML reads `1e-3` as a string (YAML 1.1 needs a decimal point), so a value that
looks like a number becomes a float.
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.paths import CONFIGS_DIR, resolve

DEFAULT_FILES = ("data.yaml", "render.yaml", "backbone.yaml", "train.yaml")

_MISSING = object()


def load_yaml(path: str | Path) -> dict:
    """Read one YAML file. A relative name is looked up in `configs/` first, then at the repository root."""
    path = Path(path)
    if not path.is_absolute():
        path = CONFIGS_DIR / path if (CONFIGS_DIR / path).exists() else resolve(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: `override` merged into `base`. Nested dicts merge; other values replace."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_value(cfg: dict, dotted_key: str, default: Any = _MISSING) -> Any:
    """Read a nested value with a dotted key, for example `train.loss.temperature`."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"config has no key {dotted_key!r}")
            return default
        node = node[part]
    return node


def set_value(cfg: dict, dotted_key: str, value: Any) -> None:
    """Write a nested value with a dotted key. Missing intermediate sections are made."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_value(text: str) -> Any:
    """Read the text of an override value as YAML; a text that looks like a number becomes a float."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_override(text: str) -> tuple[str, Any]:
    """Split `section.key=value` into the key and the parsed value."""
    if "=" not in text:
        raise ValueError(f"override must look like section.key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"override has an empty key: {text!r}")
    return key, parse_value(raw.strip())


def load_config(files: Iterable[str | Path] = DEFAULT_FILES, overrides: Iterable[str] = ()) -> dict:
    """Merge the YAML files in order, then apply the overrides in order."""
    cfg: dict = {}
    for file in files:
        cfg = deep_merge(cfg, load_yaml(file))
    for text in overrides:
        key, value = parse_override(text)
        set_value(cfg, key, value)
    return cfg


def save_config(cfg: dict, path: str | Path) -> Path:
    """Write the configuration as YAML (keys in their original order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def add_config_args(parser: argparse.ArgumentParser, default_files: Iterable[str] = DEFAULT_FILES) -> None:
    """Add `--config FILE [FILE ...]` and repeatable `--set section.key=value` to a script."""
    parser.add_argument(
        "--config",
        nargs="+",
        default=list(default_files),
        help="YAML files, merged in order (default: %(default)s)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting, for example --set train.epochs=5 (repeatable)",
    )


def config_from_args(args: argparse.Namespace) -> dict:
    """Build the configuration from the arguments that `add_config_args` added."""
    return load_config(args.config, args.overrides)


if __name__ == "__main__":
    # Stand-alone check: print the merged default configuration, with any overrides.
    _parser = argparse.ArgumentParser(description="Print the merged v2 configuration")
    add_config_args(_parser)
    _cfg = config_from_args(_parser.parse_args())
    print(yaml.safe_dump(_cfg, sort_keys=False, default_flow_style=False))
