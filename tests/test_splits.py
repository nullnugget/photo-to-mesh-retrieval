"""Tests for src/data/splits.py.

The synthetic tests need no dataset. The test marked "data" uses the real Pix3D annotations
and is skipped when the dataset is missing.
"""

import random

import pytest

from src import paths
from src.data.splits import category_folds, mesh_folds, seen_meshes_split, unseen_categories_split, unseen_meshes_split


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
