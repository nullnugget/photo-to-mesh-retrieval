"""Tests for src/config.py and src/paths.py."""

import argparse

import pytest

from src import paths
from src.config import (
    DEFAULT_FILES,
    add_config_args,
    config_from_args,
    deep_merge,
    get_value,
    load_config,
    parse_override,
    save_config,
    set_value,
)


def test_default_files_merge_into_four_sections():
    cfg = load_config()
    assert list(cfg) == ["data", "render", "backbone", "train"]


@pytest.mark.parametrize(
    "text, key, value",
    [
        ("train.optimizer.lr=1e-3", "train.optimizer.lr", 0.001),
        ("train.epochs=5", "train.epochs", 5),
        ("data.datasets=[pix3d,abo]", "data.datasets", ["pix3d", "abo"]),
        ("render.lighting=L3", "render.lighting", "L3"),
        ("train.augment.mask_noise=false", "train.augment.mask_noise", False),
        ("data.abo.max_models_per_class=null", "data.abo.max_models_per_class", None),
    ],
)
def test_override_values_get_the_right_type(text, key, value):
    k, v = parse_override(text)
    assert k == key
    assert v == value and type(v) is type(value)


def test_override_without_equals_sign_is_an_error():
    with pytest.raises(ValueError):
        parse_override("train.epochs")


def test_command_line_overrides_apply_in_order():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(["--set", "train.epochs=5", "--set", "train.epochs=7", "--set", "new.section.key=1"])
    cfg = config_from_args(args)
    assert get_value(cfg, "train.epochs") == 7
    assert get_value(cfg, "new.section.key") == 1
    assert args.config == list(DEFAULT_FILES)


def test_deep_merge_keeps_other_keys_and_does_not_change_inputs():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_get_and_set_dotted_keys():
    cfg = {}
    set_value(cfg, "x.y.z", 4)
    assert get_value(cfg, "x.y.z") == 4
    assert get_value(cfg, "x.missing", default=None) is None
    with pytest.raises(KeyError):
        get_value(cfg, "x.missing")


def test_save_and_load_round_trip(tmp_path):
    cfg = load_config(overrides=["train.epochs=3"])
    path = save_config(cfg, tmp_path / "config.yaml")
    assert load_config([path]) == cfg


def test_abo_class_rules_give_27_classes():
    rules = load_config()["data"]["abo"]["class_rules"]
    assert len({r["class"] for r in rules}) == 27


def test_backbone_settings_point_to_known_entries():
    bb = load_config()["backbone"]
    assert bb["name"] in bb["available"]
    for entry in bb["available"].values():
        assert set(entry) >= {"hub", "weights", "dim", "patch_size"}
        owner_repo, commit = entry["hub"].split(":")
        assert owner_repo.startswith("facebookresearch/") and len(commit) == 40  # pinned to a full commit hash


def test_paths_are_absolute_and_inside_the_repository():
    assert paths.ROOT.is_absolute()
    assert paths.features_file("dinov3_vitb16", "renders") == paths.FEATURES_DIR / "dinov3_vitb16_renders.npz"
    with pytest.raises(ValueError):
        paths.features_file("dinov3_vitb16", "photos")
    assert paths.resolve("configs/data.yaml") == paths.ROOT / "configs" / "data.yaml"
