"""Tests for src/data/splits.py.

The synthetic tests need no dataset. The tests marked "data" use the real Pix3D annotations
and are skipped when the dataset is missing.
"""

import random

import pytest

from src.data.splits import image_level_split, leave_categories_out_split, unseen_shape_split


def _synthetic_queries():
    """3 categories, 10 shapes each, 6 photos per shape (180 photos)."""
    rows = []
    for category in ["chair", "sofa", "table"]:
        for s in range(10):
            for p in range(6):
                rows.append({"query_id": f"x/{category}/{s}_{p}", "category": category, "shape_id": f"x/{category}/s{s}"})
    return rows


def _all_ids(split):
    return [qid for part in split["parts"].values() for qid in part]


def _assert_partition(split, queries):
    """Every photo is in exactly one part."""
    ids = _all_ids(split)
    assert len(ids) == len(set(ids)) == len(queries)


def test_image_level_split_matches_the_v1_procedure():
    queries = _synthetic_queries()
    order = [q["query_id"] for q in queries]
    split = image_level_split(order, queries, seed=7, v1_holdout_fraction=0.15)
    expected = list(order)
    random.seed(7)  # the v1 code used the global random module
    random.shuffle(expected)
    assert sorted(expected[int(len(expected) * 0.85) :]) == split["parts"]["test"]
    _assert_partition(split, queries)


def test_unavailable_photos_leave_the_image_level_split():
    queries = _synthetic_queries()
    order = [q["query_id"] for q in queries] + ["x/chair/broken"]  # in the annotations, not prepared
    split = image_level_split(order, queries)
    assert "x/chair/broken" not in _all_ids(split)
    _assert_partition(split, queries)


def test_validation_contains_every_category():
    queries = _synthetic_queries()
    split = image_level_split([q["query_id"] for q in queries], queries)
    val_categories = {qid.split("/")[1] for qid in split["parts"]["val"]}
    assert val_categories == {"chair", "sofa", "table"}


def test_unseen_shapes_share_no_shape_between_parts():
    queries = _synthetic_queries()
    split = unseen_shape_split(queries)
    shape_of = {q["query_id"]: q["shape_id"] for q in queries}
    shapes = {part: {shape_of[qid] for qid in ids} for part, ids in split["parts"].items()}
    assert not (shapes["train"] & shapes["val"]) and not (shapes["train"] & shapes["test"]) and not (shapes["val"] & shapes["test"])
    # 10 shapes per category: 2 test, 1 val, 7 train
    assert len(shapes["test"]) == 6 and len(shapes["val"]) == 3
    _assert_partition(split, queries)


def test_leave_categories_out_puts_whole_categories_in_test():
    queries = _synthetic_queries()
    split = leave_categories_out_split(queries, ["table"])
    assert {qid.split("/")[1] for qid in split["parts"]["test"]} == {"table"}
    assert all(qid.split("/")[1] != "table" for qid in split["parts"]["train"] + split["parts"]["val"])
    _assert_partition(split, queries)
    with pytest.raises(ValueError):
        leave_categories_out_split(queries, ["lamp"])


def test_splits_are_repeatable():
    queries = _synthetic_queries()
    assert unseen_shape_split(queries, seed=3) == unseen_shape_split(queries, seed=3)
    assert unseen_shape_split(queries, seed=3) != unseen_shape_split(queries, seed=4)


@pytest.mark.data
def test_real_image_level_test_part_is_the_v1_evaluation_set():
    from src.data.pix3d import load_annotations, photo_records, query_id_of

    annotations = load_annotations()
    broken = {"pix3d/table/0045", "pix3d/table/1749"}  # masks belong to other photos
    queries = [{"query_id": p.query_id, "category": p.category} for p in photo_records(annotations) if p.query_id not in broken]
    split = image_level_split([query_id_of(e) for e in annotations], queries)
    assert split["counts"]["test"] == 1510  # the number of queries that v1 evaluated
    assert sum(split["counts"].values()) == 10067
