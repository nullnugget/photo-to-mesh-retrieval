"""Tests for src/data/splits.py.

The synthetic tests need no dataset. The test marked "data" uses the real Pix3D annotations
and is skipped when the dataset is missing.
"""

import random

import pytest

from src import paths
from src.data.splits import (category_folds, left_out_categories, mesh_folds, seen_meshes_split, unseen_categories_split,
                              unseen_meshes_split)


def _synthetic_queries():
    """3 categories, 10 meshes each, 6 photos per mesh (180 photos)."""
    rows = []
    for category in ["chair", "sofa", "table"]:
        for m in range(10):
            for p in range(6):
                rows.append({"query_id": f"x/{category}/{m}_{p}", "category": category, "mesh_id": f"x/{category}/m{m}"})
    return rows


def _all_ids(split):
    return [qid for part in split["parts"].values() for qid in part]


def _assert_partition(split, queries):
    """Every photo is in exactly one part."""
    ids = _all_ids(split)
    assert len(ids) == len(set(ids)) == len(queries)


def test_seen_meshes_uses_the_fixed_test_part():
    queries = _synthetic_queries()
    order = [q["query_id"] for q in queries]
    random.seed(7)  # the v1 code used the global random module
    random.shuffle(order)
    fixed = order[int(len(order) * 0.85) :]
    split = seen_meshes_split(queries, seed=7, fixed_test=fixed)
    assert split["parts"]["test"] == sorted(fixed)
    _assert_partition(split, queries)


def test_fixed_test_ids_without_a_query_are_ignored():
    queries = _synthetic_queries()
    split = seen_meshes_split(queries, fixed_test=["x/chair/0_0", "x/chair/broken"])  # "broken" has no query image
    assert split["parts"]["test"] == ["x/chair/0_0"]
    _assert_partition(split, queries)


def test_seen_meshes_without_fixed_test_is_stratified():
    queries = _synthetic_queries()
    split = seen_meshes_split(queries, test_fraction=0.15, val_fraction=0.10)
    for part in ("test", "val"):
        categories = {qid.split("/")[1] for qid in split["parts"][part]}
        assert categories == {"chair", "sofa", "table"}
    assert split["counts"]["test"] == 3 * 9  # 15 % of 60 photos per category
    _assert_partition(split, queries)


def test_unseen_meshes_share_no_mesh_between_parts():
    queries = _synthetic_queries()
    split = unseen_meshes_split(queries)
    mesh_of = {q["query_id"]: q["mesh_id"] for q in queries}
    meshes = {part: {mesh_of[qid] for qid in ids} for part, ids in split["parts"].items()}
    assert not (meshes["train"] & meshes["val"]) and not (meshes["train"] & meshes["test"]) and not (meshes["val"] & meshes["test"])
    # 10 meshes per category: 2 test, 1 val, 7 train
    assert len(meshes["test"]) == 6 and len(meshes["val"]) == 3
    _assert_partition(split, queries)


def test_unseen_categories_puts_whole_categories_in_test():
    queries = _synthetic_queries()
    split = unseen_categories_split(queries, ["table"])
    assert {qid.split("/")[1] for qid in split["parts"]["test"]} == {"table"}
    assert all(qid.split("/")[1] != "table" for qid in split["parts"]["train"] + split["parts"]["val"])
    _assert_partition(split, queries)
    with pytest.raises(ValueError):
        unseen_categories_split(queries, ["lamp"])


def test_splits_are_repeatable():
    queries = _synthetic_queries()
    assert unseen_meshes_split(queries, seed=3) == unseen_meshes_split(queries, seed=3)
    assert unseen_meshes_split(queries, seed=3) != unseen_meshes_split(queries, seed=4)
    assert seen_meshes_split(queries, seed=3) == seen_meshes_split(queries, seed=3)


@pytest.mark.data
def test_real_seen_meshes_test_part_is_the_v1_evaluation_set():
    from src.data.pix3d import load_annotations, photo_records, v1_test_query_ids

    annotations = load_annotations()
    broken = {"pix3d/table/0045", "pix3d/table/1749"}  # masks belong to other photos
    queries = [{"query_id": p.query_id, "category": p.category} for p in photo_records(annotations) if p.query_id not in broken]
    split = seen_meshes_split(queries, fixed_test=v1_test_query_ids(annotations))
    assert split["counts"]["test"] == 1510  # the number of queries that v1 evaluated
    assert sum(split["counts"].values()) == 10067


def test_folds_use_every_non_test_mesh_once_and_keep_the_test_part():
    queries = _synthetic_queries()
    base = unseen_meshes_split(queries, seed=1)
    folds = mesh_folds(base, queries, folds=3, seed=1)
    mesh_of = {q["query_id"]: q["mesh_id"] for q in queries}
    val_photos = [qid for fold in folds for qid in fold["parts"]["val"]]
    assert sorted(val_photos) == sorted(base["parts"]["train"] + base["parts"]["val"])  # every non-test photo once
    for fold in folds:
        _assert_partition(fold, queries)
        assert fold["parts"]["test"] == base["parts"]["test"]
        assert not {mesh_of[q] for q in fold["parts"]["val"]} & {mesh_of[q] for q in fold["parts"]["train"]}  # by mesh
        assert {q.split("/")[1] for q in fold["parts"]["val"]} == {"chair", "sofa", "table"}  # every category in every fold
    assert [f["name"] for f in folds] == ["unseen_meshes_fold1", "unseen_meshes_fold2", "unseen_meshes_fold3"]
    assert mesh_folds(base, queries, folds=3, seed=1) == folds  # repeatable


def test_folds_need_a_split_by_mesh():
    queries = _synthetic_queries()
    with pytest.raises(ValueError, match="by mesh"):
        mesh_folds(seen_meshes_split(queries, seed=1), queries, folds=3)


def test_a_fold_has_a_split_file_name():
    assert paths.split_file("pix3d", "unseen_meshes_fold2").name == "unseen_meshes_fold2.json"
    with pytest.raises(ValueError):
        paths.split_file("pix3d", "unseen_meshes_foldx")


def test_category_folds_leave_one_category_out():
    queries = _synthetic_queries()
    base = unseen_categories_split(queries, held_out=["table"], seed=1)
    folds = category_folds(base, queries)
    category_of = {q["query_id"]: q["category"] for q in queries}
    assert [f["held_out_category"] for f in folds] == ["chair", "sofa"]
    for fold in folds:
        _assert_partition(fold, queries)
        assert fold["parts"]["test"] == base["parts"]["test"]
        assert {category_of[q] for q in fold["parts"]["val"]} == {fold["held_out_category"]}
        assert fold["held_out_category"] not in {category_of[q] for q in fold["parts"]["train"]}
    with pytest.raises(ValueError, match="by category"):
        category_folds(unseen_meshes_split(queries, seed=1), queries)


def test_category_folds_can_follow_the_categories_of_another_dataset():
    # A joint training: the folds of the first dataset are (bed, chair); this dataset has no bed.
    queries = _synthetic_queries()
    base = unseen_categories_split(queries, held_out=["table"], seed=1)
    folds = category_folds(base, queries, categories=["bed", "chair"])
    category_of = {q["query_id"]: q["category"] for q in queries}
    assert [f["held_out_category"] for f in folds] == ["bed", "chair"]
    assert folds[0]["parts"]["val"] == []  # no photo of the category: an empty val part, all in train
    assert {category_of[q] for q in folds[0]["parts"]["train"]} == {"chair", "sofa"}
    assert {category_of[q] for q in folds[1]["parts"]["val"]} == {"chair"}
    for fold in folds:
        _assert_partition(fold, queries)


def test_left_out_categories_add_the_near_relatives():
    relatives = {"bookcase": ["shelf", "tv_stand"], "chair": ["stool", "bench"], "sofa": ["ottoman", "bench"]}
    assert left_out_categories({"bookcase", "chair"}, relatives) == {"bookcase", "shelf", "tv_stand", "chair", "stool", "bench"}
    assert left_out_categories({"bed"}, relatives) == {"bed"}  # no relatives
    assert left_out_categories(set(), relatives) == set()
    assert left_out_categories({"chair"}, None) == {"chair"}


@pytest.mark.data
def test_an_unseen_categories_training_leaves_out_the_near_relatives():
    from src.config import load_config
    from src.data.datasets import training_photos
    from src.pipeline.train_heads import held_out_categories

    assert held_out_categories(["pix3d"], "unseen_categories") == {"bookcase", "wardrobe"}
    assert held_out_categories(["pix3d"], "unseen_categories_fold2") == {"bookcase", "wardrobe", "chair"}
    assert held_out_categories(["pix3d"], "seen_meshes") == set()
    held = held_out_categories(["pix3d", "abo"], "unseen_categories")
    assert held == {"bookcase", "wardrobe", "lamp", "ceiling_light", "planter", "vase"}
    cfg = load_config()
    leave_out = left_out_categories(held, cfg["data"]["splits"]["unseen_categories"]["relatives"])
    assert {"shelf", "tv_stand", "cabinet", "dresser"} <= leave_out
    photos, report = training_photos(["pix3d", "abo"], "unseen_categories", "train", cfg["train"]["query_images"], leave_out)
    assert not {p.category for p in photos} & leave_out
    assert set(report["left_out_categories"]) == {"abo/shelf", "abo/tv_stand", "abo/cabinet", "abo/dresser"}
    # Pix3D alone loses nothing: no Pix3D category is a near relative.
    pix3d_only, _ = training_photos(["pix3d"], "unseen_categories", "train", cfg["train"]["query_images"], leave_out)
    unchanged, _ = training_photos(["pix3d"], "unseen_categories", "train", cfg["train"]["query_images"])
    assert len(pix3d_only) == len(unchanged)


def test_carry_over_keeps_parts_and_moves_a_split_mesh_out_of_test():
    from src.data.splits import carry_over_parts
    old = {"name": "unseen_meshes", "description": "d", "seed": 1,
           "parts": {"train": ["abo/chair/p2", "abo/sofa/p3"], "val": ["abo/sofa/p4"], "test": ["abo/chair/p1", "abo/table/p6"]}}
    # After a merge: p1 (test) and p2 (train) show the same entry m1; p3's class changed (chair → sofa did not matter);
    # p5 is new; p6 is alone in the test part.
    queries = [{"query_id": "abo/chair/p1", "mesh_id": "m1"}, {"query_id": "abo/chair/p2", "mesh_id": "m1"},
               {"query_id": "abo/chair/p3", "mesh_id": "m2"}, {"query_id": "abo/sofa/p4", "mesh_id": "m3"},
               {"query_id": "abo/sofa/p5", "mesh_id": "m4"}, {"query_id": "abo/table/p6", "mesh_id": "m5"}]
    fresh = {"name": "unseen_meshes", "description": "fresh", "seed": 1,
             "parts": {"train": ["abo/chair/p1", "abo/chair/p2", "abo/chair/p3"], "val": ["abo/sofa/p5"], "test": ["abo/sofa/p4", "abo/table/p6"]}}
    kept = carry_over_parts(old, fresh, queries, by_mesh=True)
    assert kept["parts"]["test"] == ["abo/table/p6"]  # p1 left the test part with its mesh; nothing moved in
    assert set(kept["parts"]["train"]) == {"abo/chair/p1", "abo/chair/p2", "abo/chair/p3"}
    assert kept["parts"]["val"] == ["abo/sofa/p4", "abo/sofa/p5"]  # p4 kept val (fresh said test); p5 is new: fresh part
    photo_level = carry_over_parts(old, fresh, queries, by_mesh=False)
    assert photo_level["parts"]["test"] == ["abo/chair/p1", "abo/table/p6"]
