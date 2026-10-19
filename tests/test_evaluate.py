"""Tests for src/retrieval/evaluate.py, the new metrics, the test-free photos, and the test lock."""

import json

import numpy as np
import pytest

from src import paths
from src.config import load_config
from src.data import splits  # through the module: pytest would collect a name that starts with "test_"
from src.pipeline import run_evaluation
from src.retrieval import metrics
from src.retrieval.evaluate import Gallery, Queries, evaluate

# A gallery of three meshes with one view each (2 numbers per vector), and four queries.
#   chair_a (1, 0)   chair_b (0.8, 0.6)   table_a (0, 1)
GALLERY = Gallery(
    views=np.array([[[1.0, 0.0]], [[0.8, 0.6]], [[0.0, 1.0]]]),
    mesh_ids=np.array(["chair_a", "chair_b", "table_a"]),
    categories=np.array(["chair", "chair", "table"]),
)
# q1 finds chair_a first (right). q2 wants chair_b but chair_a is first (wrong mesh, right category).
# q3 wants table_a but chair_b is first (wrong category); inside the tables it is first.
# q4 wants table_a and finds it first.
QUERIES = Queries(
    vectors=np.array([[1.0, 0.0], [0.995, 0.1], [0.75, 0.66], [0.1, 0.995]]),
    query_ids=np.array(["q1", "q2", "q3", "q4"]),
    mesh_ids=np.array(["chair_a", "chair_b", "table_a", "table_a"]),
    categories=np.array(["chair", "chair", "table", "table"]),
)
FLAGS = {
    "q1": {"truncated": False, "occluded": False, "slightly_occluded": False},
    "q2": {"truncated": True, "occluded": False, "slightly_occluded": False},
    "q3": {"truncated": False, "occluded": True, "slightly_occluded": False},
    "q4": {"truncated": False, "occluded": False, "slightly_occluded": False},
}


def test_metrics_of_a_hand_made_example():
    result, rows = evaluate(QUERIES, GALLERY, "view", flags=FLAGS)
    assert [row["rank"] for row in rows] == [1, 2, 3, 1]
    assert [row["rank_inside_category"] for row in rows] == [1, 2, 1, 1]
    assert [row["first_mesh_id"] for row in rows] == ["chair_a", "chair_a", "chair_b", "table_a"]
    assert result["R@1"] == 0.5 and result["R@5"] == 1.0 and result["photos"] == 4 and result["gallery_meshes"] == 3
    assert result["cat@1"] == 0.75  # only q3 picks a wrong category first
    assert result["R@1_inside_category"] == 0.75  # q3 is right among the tables; q2 stays wrong
    assert result["per_category"] == {"chair": {"queries": 2, "R@1": 0.5}, "table": {"queries": 2, "R@1": 0.5}}
    assert result["macro_R@1"] == 0.5
    assert result["subsets"] == {"clean": {"queries": 2, "R@1": 1.0}, "truncated": {"queries": 1, "R@1": 0.0},
                                 "occluded": {"queries": 1, "R@1": 0.0}}


def test_photos_without_annotations_have_no_subsets():
    unknown = {q: {"truncated": None, "occluded": None, "slightly_occluded": None} for q in FLAGS}
    assert evaluate(QUERIES, GALLERY, "view", flags=unknown)[0]["subsets"] == {}


def test_a_correct_mesh_outside_the_gallery_is_an_error():
    gallery = Gallery(GALLERY.views[:2], GALLERY.mesh_ids[:2], GALLERY.categories[:2])
    with pytest.raises(KeyError):
        evaluate(QUERIES, gallery, "view")


def test_macro_recall_gives_every_group_the_same_weight():
    ranks = np.array([1, 1, 1, 2])
    groups = np.array(["chair", "chair", "chair", "sofa"])
    assert metrics.recall_at_k(ranks, 1) == 0.75
    assert metrics.macro_recall(ranks, groups) == 0.5  # chair 1.0, sofa 0.0


def _write_splits(splits_dir, parts_by_split):
    for name, parts in parts_by_split.items():
        path = paths.split_file("toy", name, splits_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name, "parts": parts}))


def test_test_free_photos_are_in_no_test_part(tmp_path):
    _write_splits(tmp_path, {
        "seen_meshes": {"train": ["a", "b", "c"], "val": ["d"], "test": ["e"]},
        "unseen_meshes": {"train": ["a", "e"], "val": ["b"], "test": ["c", "d"]},
        "unseen_categories": {"train": ["b", "c", "d", "e"], "val": [], "test": ["a"]},
    })
    assert splits.test_free_query_ids("toy", tmp_path) == ["b"]


def test_test_free_photos_need_splits_of_the_same_photos(tmp_path):
    _write_splits(tmp_path, {
        "seen_meshes": {"train": ["a"], "val": [], "test": ["b"]},
        "unseen_meshes": {"train": ["a"], "val": [], "test": ["c"]},
        "unseen_categories": {"train": ["a"], "val": [], "test": ["b"]},
    })
    with pytest.raises(ValueError):
        splits.test_free_query_ids("toy", tmp_path)


def test_the_test_part_is_locked_and_the_photo_groups_are_checked():
    cfg = load_config()
    with pytest.raises(run_evaluation.TestPhotosLocked):
        run_evaluation.run("pix3d", cfg, photos="test", split="seen_meshes")
    with pytest.raises(ValueError):
        run_evaluation.photo_ids("pix3d", "test_free", "seen_meshes")  # test-free photos belong to no split
    with pytest.raises(ValueError):
        run_evaluation.photo_ids("pix3d", "val", None)  # a part needs its split
