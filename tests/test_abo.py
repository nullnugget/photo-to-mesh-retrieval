"""Tests for src/data/abo.py, on small synthetic metadata (no download needed)."""

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
