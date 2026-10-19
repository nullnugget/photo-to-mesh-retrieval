"""Tests for src/models/feature_store.py, the recipe combination, and the CPU parts of the feature step."""

import os

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.models.backbones import FeatureExtractor, combine_recipe, object_masks
from src.models.feature_store import QUERIES, RENDERS, FeatureTable, check_table, read_features, write_features
from src.pipeline import extract_features


def _unit_rows(rng, *shape):
    vectors = rng.normal(size=shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _queries(rows=4, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    parts = {"cls": _unit_rows(rng, rows, dim).astype(np.float16), "masked_mean": _unit_rows(rng, rows, dim).astype(np.float16)}
    labels = {
        "query_id": np.array([f"pix3d/chair/{i:04d}" for i in range(rows)]),
        "mesh_id": np.array(["pix3d/chair/IKEA_POANG_2"] * rows),
        "category": np.array(["chair"] * rows),
    }
    return FeatureTable(QUERIES, parts, labels, {"backbone": "dinov3_vitb16", "feature_set": "queries_sam"})


def _renders(meshes=3, views=25, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    parts = {name: _unit_rows(rng, meshes, views, dim).astype(np.float16) for name in ("cls", "masked_mean")}
    labels = {"mesh_id": np.array([f"abo/lamp/B{i}" for i in range(meshes)]), "category": np.array(["lamp"] * meshes)}
    return FeatureTable(RENDERS, parts, labels, {"views": views})


@pytest.mark.parametrize("make", [_queries, _renders])
def test_a_written_file_reads_back_identically(tmp_path, make):
    table = make()
    path = write_features(table, tmp_path / "features" / "set.npz")
    read = read_features(path)
    assert read.kind == table.kind and read.info == table.info
    for name in table.parts:
        assert read.parts[name].dtype == np.float16 and np.array_equal(read.parts[name], table.parts[name])
    for name in table.labels:
        assert read.labels[name].tolist() == table.labels[name].tolist()
    assert not list(tmp_path.rglob("*.part"))  # the temporary file was renamed


def test_float32_vectors_are_stored_as_float16(tmp_path):
    table = _queries()
    table.parts = {name: values.astype(np.float32) + np.float32(1e-5) for name, values in table.parts.items()}
    read = read_features(write_features(table, tmp_path / "set.npz"))
    assert np.array_equal(read.parts["cls"], table.parts["cls"].astype(np.float16))


def test_inconsistent_tables_are_not_written(tmp_path):
    repeated = _queries()
    repeated.labels["query_id"][1] = repeated.labels["query_id"][0]
    short = _queries()
    short.labels["mesh_id"] = short.labels["mesh_id"][:-1]
    no_category = _queries()
    del no_category.labels["category"]
    flat_renders = _renders()
    flat_renders.parts = {name: values[:, 0] for name, values in flat_renders.parts.items()}  # (meshes, dim): views missing
    one_part = _queries()
    del one_part.parts["masked_mean"]
    for table in (repeated, short, no_category, flat_renders, one_part):
        with pytest.raises(ValueError):
            write_features(table, tmp_path / "set.npz")
    assert not (tmp_path / "set.npz").exists()


def test_select_returns_the_rows_in_the_given_order():
    table = _queries(rows=5)
    ids = [table.ids[3], table.ids[0]]
    selected = table.select(ids)
    assert selected.ids.tolist() == ids
    assert np.array_equal(selected.parts["cls"], table.parts["cls"][[3, 0]])
    check_table(selected)
    with pytest.raises(KeyError):
        table.select(["pix3d/chair/9999"])


def test_recipes_are_unit_length_and_joined_with_equal_weight():
    table = _renders()
    for recipe, dim in (("cls", 8), ("masked_mean", 8), ("cls_masked_mean", 16)):
        vectors = table.vectors(recipe)
        assert vectors.dtype == np.float32 and vectors.shape == (3, 25, dim)
        assert np.allclose(np.linalg.norm(vectors, axis=-1), 1.0, atol=1e-6)
    joined = combine_recipe({"cls": np.array([[1.0, 0.0]]), "masked_mean": np.array([[0.0, 2.0]])}, "cls_masked_mean")
    assert np.allclose(joined, [[1, 0, 0, 1]] / np.sqrt(2))  # each part scaled to length 1 first
    with pytest.raises(ValueError):
        combine_recipe(table.parts, "mean")


def test_mini_sample_keeps_the_meshes_of_its_queries(monkeypatch):
    meshes = [{"mesh_id": f"m{i}", "dataset": "abo", "category": "lamp" if i < 4 else "vase", "num_photos": 0 if i == 0 else 5}
              for i in range(6)]
    queries = [{"query_id": f"q{m['mesh_id']}_{j}", "dataset": "abo", "mesh_id": m["mesh_id"]} for m in meshes[1:] for j in range(5)]
    monkeypatch.setattr(extract_features, "read_table", lambda path: meshes if path.name == "meshes.csv" else queries)
    mini_meshes, mini_queries = extract_features.catalog_rows("abo", mini=True)
    assert [m["mesh_id"] for m in mini_meshes] == ["m1", "m2", "m4", "m5"]  # 2 per category, only meshes with photos
    assert len(mini_queries) == 4 * extract_features.MINI_PHOTOS_PER_MESH
    assert {q["mesh_id"] for q in mini_queries} == {"m1", "m2", "m4", "m5"}
    full_meshes, full_queries = extract_features.catalog_rows("abo", mini=False)
    assert len(full_meshes) == 6 and len(full_queries) == 25


def test_sam_query_images_are_made_again_when_the_mask_is_newer(tmp_path, monkeypatch):
    photo, mask = tmp_path / "0001.jpg", tmp_path / "masks" / "0001.png"
    mask.parent.mkdir()
    Image.fromarray(np.full((40, 60, 3), 200, dtype=np.uint8)).save(photo)
    mask_pixels = np.zeros((40, 60), dtype=np.uint8)
    mask_pixels[10:30, 20:40] = 255
    Image.fromarray(mask_pixels).save(mask)
    rows = [
        {"query_id": "pix3d/chair/0001", "dataset": "pix3d", "category": "chair", "photo_path": str(photo), "query_path": "q/chair/0001.png"},
        {"query_id": "pix3d/chair/0002", "dataset": "pix3d", "category": "chair", "photo_path": str(photo), "query_path": "q/chair/0002.png"},
    ]
    monkeypatch.setattr(extract_features, "sam_mask_file", lambda row: mask if row["query_id"].endswith("0001") else tmp_path / "none.png")
    data_cfg = {"background_rgb": [128, 128, 128], "crop_margin": 0.0, "image_size": 32}
    root = tmp_path / "out"

    made, left_out = extract_features.make_sam_query_images(rows, root, data_cfg, overwrite=False)
    target = extract_features.sam_query_file(root, rows[0])
    assert [r["query_id"] for r in made] == ["pix3d/chair/0001"] and target.exists()
    assert left_out == [{"query_id": "pix3d/chair/0002", "message": "no SAM mask"}]
    assert target == root / "pix3d_sam_queries" / "chair" / "0001.png"

    os.utime(target, (1, 1))  # the image is now older than the mask: it must be made again
    extract_features.make_sam_query_images(rows[:1], root, data_cfg, overwrite=False)
    assert target.stat().st_mtime > mask.stat().st_mtime - 1
    before = target.stat().st_mtime_ns
    extract_features.make_sam_query_images(rows[:1], root, data_cfg, overwrite=False)  # up to date: skipped
    assert target.stat().st_mtime_ns == before


def test_feature_sets_of_each_dataset():
    assert extract_features.feature_sets("pix3d") == ["renders", "queries_reference", "queries_sam"]
    assert extract_features.feature_sets("abo") == ["renders", "queries_sam"]


@pytest.mark.gpu
def test_extract_equals_the_combined_parts_and_repeats_exactly():
    extractor = FeatureExtractor(load_config(["backbone.yaml"])["backbone"], recipe="cls_masked_mean")
    images = np.full((3, 224, 224, 3), 128, dtype=np.uint8)
    images[:, 60:170, 80:150] = np.random.default_rng(0).integers(0, 255, size=(3, 110, 70, 3), dtype=np.uint8)
    masks = object_masks(images, (128, 128, 128))
    parts = extractor.extract_parts(images, masks)
    assert np.allclose(extractor.extract(images, masks), combine_recipe(parts, "cls_masked_mean"), atol=1e-6)
    again = extractor.extract_parts(images, masks)
    assert all(np.abs(parts[name] - again[name]).max() < 1e-3 for name in parts)  # the same image gives the same features
