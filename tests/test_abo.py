"""Tests for src/data/abo.py, on small synthetic metadata (no download needed)."""

import pytest

from src.data import abo

RULES = [
    {"product_type": "CHAIR", "class": "chair"},
    {"product_type": "HOME", "leaf": "Space Heaters", "class": "space_heater"},
]


def test_class_rules_use_type_and_leaf():
    assert abo.class_of("CHAIR", "", RULES) == "chair"
    assert abo.class_of("HOME", "Space Heaters", RULES) == "space_heater"
    assert abo.class_of("HOME", "Picture Frames", RULES) is None  # leaf does not match
    assert abo.class_of("RUG", "", RULES) is None  # no rule


def _model(path):
    return {"glb_path": path, "num_vertices": 10, "num_faces": 12, "extent": (1.0, 2.0, 3.0)}


def test_select_keeps_shared_photos_once():
    models = {"B1": _model("1/B1.glb"), "B2": _model("2/B2.glb"), "B3": _model("3/B3.glb")}
    listings = {
        "B1": {"product_type": "CHAIR", "leaf": "", "image_ids": ["img_a", "img_shared"]},
        "B2": {"product_type": "CHAIR", "leaf": "", "image_ids": ["img_shared", "img_missing"]},
        "B3": {"product_type": "RUG", "leaf": "", "image_ids": ["img_rug"]},
    }
    images = {i: {"path": f"x/{i}.jpg", "width": 100, "height": 100} for i in ["img_a", "img_shared", "img_rug"]}
    meshes, photos = abo.select(models, listings, images, RULES)
    assert [m.mesh_id for m in meshes] == ["abo/chair/B1", "abo/chair/B2"]  # the rug has no class
    assert meshes[1].image_ids == ["img_shared"]  # a photo id missing from images.csv is left out
    shared = next(p for p in photos if p.image_id == "img_shared")
    assert shared.mesh_ids == ["abo/chair/B1", "abo/chair/B2"] and len(photos) == 2


def test_cap_per_class_and_manifest_round_trip(tmp_path):
    models = {f"B{i}": _model(f"{i}/B{i}.glb") for i in range(5)}
    listings = {f"B{i}": {"product_type": "CHAIR", "leaf": "Chairs", "image_ids": [f"i{i}"]} for i in range(5)}
    images = {f"i{i}": {"path": f"x/i{i}.jpg", "width": 10, "height": 20} for i in range(5)}
    meshes, photos = abo.select(models, listings, images, RULES, max_models_per_class=2)
    assert [m.model_id for m in meshes] == ["B0", "B1"]
    abo.write_manifest(meshes, photos, tmp_path)
    assert abo.read_manifest(tmp_path) == (meshes, photos)


def test_class_override_wins_over_the_rules_and_keeps_amazons_label():
    models = {"B1": _model("1/B1.glb"), "B2": _model("2/B2.glb")}
    listings = {
        "B1": {"product_type": "CHAIR", "leaf": "", "image_ids": ["i1"]},  # a sofa with a wrong label
        "B2": {"product_type": "RUG", "leaf": "", "image_ids": []},  # no rule; the override adds it
    }
    images = {"i1": {"path": "x/i1.jpg", "width": 10, "height": 10}}
    rules = RULES + [{"product_type": "SOFA", "class": "sofa"}]
    meshes, photos = abo.select(models, listings, images, rules, class_overrides={"B1": "sofa", "B2": "chair"})
    assert [(m.mesh_id, m.product_type) for m in meshes] == [("abo/chair/B2", "RUG"), ("abo/sofa/B1", "CHAIR")]
    assert photos[0].category == "sofa" and photos[0].mesh_ids == ["abo/sofa/B1"]


@pytest.mark.parametrize("overrides", [{"B9": "chair"}, {"B1": "sofaa"}])
def test_class_override_with_unknown_id_or_class_is_an_error(overrides):
    models = {"B1": _model("1/B1.glb")}
    listings = {"B1": {"product_type": "CHAIR", "leaf": "", "image_ids": []}}
    with pytest.raises(ValueError):
        abo.select(models, listings, {}, RULES, class_overrides=overrides)


def test_photo_entry_needs_a_single_geometry_group():
    from src.pipeline.catalog_abo_meshes import photo_entry

    groups = {"abo/sofa/A": "abo/sofa/A", "abo/sofa/B": "abo/sofa/A", "abo/sofa/C": "abo/sofa/C"}
    photo = abo.AboImage("i1", "sofa", "x/i1.jpg", 10, 10, ["abo/sofa/A", "abo/sofa/B"])
    assert photo_entry(photo, groups) == "abo/sofa/A"  # two copies of one geometry: one entry
    photo.mesh_ids = ["abo/sofa/A", "abo/sofa/C"]
    assert photo_entry(photo, groups) is None  # two different geometries: no single answer
    photo.mesh_ids = ["abo/sofa/missing"]
    assert photo_entry(photo, groups) is None


def test_main_image_ids_and_the_photo_check_rule(tmp_path):
    import gzip
    import json

    from src.data.abo import main_image_ids
    from src.pipeline.check_photos import EXTRA, MAIN, is_left_out

    listings = [{"3dmodel_id": "M1", "main_image_id": "a", "other_image_id": ["b", "c"]},
                {"main_image_id": "x", "other_image_id": ["y"]}]  # no 3D model: not one of ours
    path = tmp_path / "listings_0.json.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(json.dumps(listing) for listing in listings) + "\n")
    assert main_image_ids([path]) == {"a"}
    assert is_left_out(EXTRA, 0.41, 0.4) and not is_left_out(EXTRA, 0.4, 0.4)
    assert not is_left_out(MAIN, 0.99, 0.4)  # a main image is never left out
