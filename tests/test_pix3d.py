"""Tests for src/data/pix3d.py and src/preprocess/queries.py on the real Pix3D data.

All tests are marked "data" and are skipped when the dataset is missing.
"""

import numpy as np
import pytest

from src.data.pix3d import load_annotations, mesh_records, photo_records, v1_test_query_ids
from src.preprocess.queries import MaskMismatchError, read_photo_and_mask

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def annotations():
    return load_annotations()


def test_counts_and_identifiers(annotations):
    photos = photo_records(annotations)
    meshes = mesh_records(annotations)
    assert len(photos) == 10069 and len(meshes) == 395
    assert len({p.query_id for p in photos}) == len(photos)  # photo ids are unique
    by_id = {m.mesh_id: m for m in meshes}
    assert "pix3d/chair/IKEA_EKTORP_1" in by_id and "pix3d/sofa/IKEA_EKTORP_1" in by_id  # same folder, two categories
    assert sum(m.num_mesh_variants > 1 for m in meshes) == 5  # the swivel chairs


def test_broken_masks_raise(annotations):
    by_id = {p.query_id: p for p in photo_records(annotations)}
    for query_id in ["pix3d/table/0045", "pix3d/table/1749"]:
        with pytest.raises(MaskMismatchError):
            read_photo_and_mask(by_id[query_id].photo_path, by_id[query_id].mask_path)


def test_palette_photo_reads_as_rgb(annotations):
    by_id = {p.query_id: p for p in photo_records(annotations)}
    image, mask = read_photo_and_mask(by_id["pix3d/desk/0389"].photo_path, by_id["pix3d/desk/0389"].mask_path)
    assert image.dtype == np.uint8 and image.shape[2] == 3 and mask.dtype == bool and mask.any()


def test_v1_test_list(annotations):
    test = v1_test_query_ids(annotations)
    assert len(test) == 1511 and "pix3d/table/0045" in test
