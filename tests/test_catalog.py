"""Tests for src/data/catalog.py. All files go into pytest's temporary folder."""

import pytest

from src.data.catalog import MESH_COLUMNS, QUERY_COLUMNS, read_table, replace_dataset_rows, write_table


def _query_row(query_id, dataset="pix3d", truncated=False):
    row = {column: "x" for column in QUERY_COLUMNS}
    row.update(query_id=query_id, dataset=dataset, truncated=truncated, occluded=True, slightly_occluded=False)
    return row


def test_round_trip_keeps_values_and_types(tmp_path):
    path = write_table([_query_row("pix3d/chair/0001", truncated=True)], tmp_path / "queries.csv", QUERY_COLUMNS)
    (row,) = read_table(path)
    assert row["query_id"] == "pix3d/chair/0001"
    assert row["truncated"] is True and row["occluded"] is True and row["slightly_occluded"] is False


def test_unknown_flags_are_read_as_none(tmp_path):
    row = _query_row("abo/chair/x")
    row.update(truncated=None, occluded=None, slightly_occluded=None)  # a dataset without these annotations
    (read,) = read_table(write_table([row], tmp_path / "queries.csv", QUERY_COLUMNS))
    assert read["truncated"] is None and read["occluded"] is None and read["slightly_occluded"] is None


def test_integer_columns_are_read_as_int(tmp_path):
    row = {column: "x" for column in MESH_COLUMNS}
    row.update(num_photos=7, num_mesh_variants=1)
    (read,) = read_table(write_table([row], tmp_path / "meshes.csv", MESH_COLUMNS))
    assert read["num_photos"] == 7 and read["num_mesh_variants"] == 1


def test_missing_or_extra_column_is_an_error(tmp_path):
    row = _query_row("pix3d/chair/0001")
    del row["mask_path"]
    with pytest.raises(KeyError):
        write_table([row], tmp_path / "q.csv", QUERY_COLUMNS)
    with pytest.raises(KeyError):
        write_table([{**_query_row("a"), "extra": 1}], tmp_path / "q.csv", QUERY_COLUMNS)


def test_replace_dataset_rows_keeps_other_datasets(tmp_path):
    path = tmp_path / "queries.csv"
    write_table([_query_row("pix3d/a"), _query_row("abo/b", dataset="abo")], path, QUERY_COLUMNS)
    replace_dataset_rows(path, [_query_row("pix3d/c"), _query_row("pix3d/d")], "pix3d", QUERY_COLUMNS)
    ids = sorted(row["query_id"] for row in read_table(path))
    assert ids == ["abo/b", "pix3d/c", "pix3d/d"]


def test_replace_rejects_rows_of_another_dataset(tmp_path):
    with pytest.raises(ValueError):
        replace_dataset_rows(tmp_path / "q.csv", [_query_row("abo/x", dataset="abo")], "pix3d", QUERY_COLUMNS)
