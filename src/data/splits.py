"""Data splits: which query photos are used for training, for validation, and for the test.

A split divides the query photos of one dataset into three parts:

- train: photos that the model learns from;
- val:   photos that select checkpoints and settings during development;
- test:  photos that give the reported results. They are never used to choose anything.

The three splits differ in what is new at test time. The names say what the test part contains:

1. "seen_meshes": can the model find a mesh that it has seen in training photos, from a new photo
   of it? Photos are split at random, so most meshes have photos in every part. This is the
   easiest case. (For Pix3D, the test part is exactly the v1 evaluation set, so v1 and v2
   results are comparable; see `src/data/pix3d.py`, `v1_test_query_ids`.)
2. "unseen_meshes": can the model find a mesh of which it has never seen a photo? Meshes are
   split within each category, and all photos of a mesh go to the same part. The model has seen
   other meshes of the same category (other chairs), but not this one.
3. "unseen_categories": can the model find meshes of a category that it has never trained on?
   All photos of the held-out categories (for example every bookcase) form the test part.

Example, the chair IKEA_POANG_2 with about 30 photos: in "seen_meshes", about 25 of its photos
are training photos and about 3 are test photos. In "unseen_meshes", all 30 photos are in one
part; if that part is the test part, no photo of this chair is used in training. In
"unseen_categories" (bookcase and wardrobe held out), chairs stay in training.

The gallery and the training. In real use, a new product's 3D mesh is added to the database
after the training, and no photo of it was used for the training. The splits copy this:

- During training, the model compares each training photo only with the meshes of the training
  photos (the "training gallery"). Meshes whose photos are all in the val or test part are not in
  the training gallery, so the model never sees them (not their photos and not their renders).
- During evaluation, the gallery contains every mesh. Thus a test photo must find its mesh among
  all meshes, also the meshes that the model has never seen.

Shared settings and the test-free photos. The three splits are three independent experiments: each
trains its own model, so a photo can be a training photo in one split and a test photo in another.
But some settings serve all three experiments at once, because the project has only one set of each:
the lighting of the renders, the backbone, the feature recipe, the segmenter, and the zero-shot
scoring. A selection also fits the photos that it is made on a little: among many settings, the
best one on some photos is partly lucky on exactly these photos. So a shared setting must not be
selected on a photo that is a test photo of any split. The "test-free photos" of a dataset are the
photos that are in no test part of any of its splits (`test_free_query_ids`). Shared settings are
selected only on them; each test part is evaluated once, after all selections are fixed.

Every split is repeatable: it depends only on the input rows and a seed. Split files are small
JSON files in `data/splits/<dataset>/` and are tracked in git, so that every run uses the same
split. A split works for any dataset: it needs only the query id, the category, and the mesh id
of each photo.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from src import paths

PARTS = ("train", "val", "test")


def _group_by(items: list[dict], key: str) -> dict[str, list[dict]]:
    """Group dicts by the value of one key. The groups are returned in sorted key order."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return dict(sorted(groups.items()))


def _stratified_sample(query_ids_by_category: dict[str, list[str]], fraction: float, rng: random.Random) -> set[str]:
    """Take about `fraction` of the photos of each category, at random.

    Each category with at least 2 photos gives at least 1 photo, so that small categories
    (for example "tool", 47 photos) still appear in the sample.
    """
    sample: set[str] = set()
    for category, query_ids in query_ids_by_category.items():
        ordered = sorted(query_ids)  # sorted first, so that the result depends only on the seed
        count = max(1, round(len(ordered) * fraction)) if len(ordered) >= 2 else 0
        sample.update(rng.sample(ordered, count))
    return sample


def _ids_by_category(rows: list[dict]) -> dict[str, list[str]]:
    """Return {category: [query ids]}, with the categories in sorted order."""
    return {c: [r["query_id"] for r in group] for c, group in _group_by(rows, "category").items()}


def _make_split(name: str, description: str, seed: int, train, val, test) -> dict:
    """Build the split dict that is saved as JSON (sorted lists, plus their sizes)."""
    parts = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    overlap = (set(parts["train"]) & set(parts["val"])) | (set(parts["train"]) & set(parts["test"])) | (
        set(parts["val"]) & set(parts["test"])
    )
    if overlap:
        raise AssertionError(f"split {name}: {len(overlap)} photos are in two parts")
    return {
        "name": name,
        "description": description,
        "seed": seed,
        "counts": {part: len(ids) for part, ids in parts.items()},
        "parts": parts,
    }


def seen_meshes_split(
    queries: list[dict],
    seed: int = 42,
    test_fraction: float = 0.15,
    val_fraction: float = 0.10,
    fixed_test: list[str] | None = None,
) -> dict:
    """Split the photos at random (stratified by category). Meshes appear in several parts.

    Args:
        queries: Catalog rows of the photos that exist as queries (query_id, category).
        seed: Seed of the random samples.
        test_fraction: Share of each category's photos that forms the test part. Not used when
            `fixed_test` is given.
        val_fraction: Share of each category's remaining (non-test) photos kept for validation.
        fixed_test: Optional list of query ids that must form the test part, for example the v1
            evaluation set of Pix3D. Ids that are not in `queries` (photos that could not be
            prepared) are ignored.

    The validation photos are always a stratified sample, so that every category is in them.
    """
    rng = random.Random(seed)
    available = {q["query_id"] for q in queries}
    if fixed_test is None:
        test = _stratified_sample(_ids_by_category(queries), test_fraction, rng)
        test_rule = f"a random {test_fraction:.0%} of each category"
    else:
        test = set(fixed_test) & available
        test_rule = "a fixed list of photos"
    remaining = [q for q in queries if q["query_id"] not in test]
    val = _stratified_sample(_ids_by_category(remaining), val_fraction, rng)
    train = [q["query_id"] for q in remaining if q["query_id"] not in val]
    return _make_split(
        "seen_meshes",
        f"Photos split at random; most meshes have photos in every part. Test part: {test_rule}.",
        seed,
        train,
        val,
        test,
    )


def unseen_meshes_split(queries: list[dict], seed: int = 42, fractions: dict | None = None) -> dict:
    """Split the meshes of each category into train, val, and test; each photo follows its mesh.

    Args:
        queries: Catalog rows (query_id, category, mesh_id).
        seed: Seed of the random mesh assignment.
        fractions: Shares of meshes, for example {"train": 0.7, "val": 0.1, "test": 0.2}.

    In every category with at least 3 meshes, val and test get at least 1 mesh each.
    """
    fractions = fractions or {"train": 0.70, "val": 0.10, "test": 0.20}
    rng = random.Random(seed)
    part_of_mesh: dict[str, str] = {}
    for category, rows in _group_by(queries, "category").items():
        meshes = sorted({row["mesh_id"] for row in rows})
        rng.shuffle(meshes)
        n = len(meshes)
        n_test = max(1, round(n * fractions["test"])) if n >= 3 else 0
        n_val = max(1, round(n * fractions["val"])) if n >= 3 else 0
        for i, mesh in enumerate(meshes):
            part_of_mesh[mesh] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    parts = {"train": [], "val": [], "test": []}
    for row in queries:
        parts[part_of_mesh[row["mesh_id"]]].append(row["query_id"])
    return _make_split(
        "unseen_meshes",
        "Meshes split within each category (about 70/10/20); all photos of a mesh are in one part.",
        seed,
        parts["train"],
        parts["val"],
        parts["test"],
    )


def unseen_categories_split(
    queries: list[dict], held_out: list[str], seed: int = 42, val_fraction: float = 0.10
) -> dict:
    """Use whole categories as the test part.

    Args:
        queries: Catalog rows (query_id, category).
        held_out: Categories whose photos form the test part, for example ["bookcase", "wardrobe"].
        seed: Seed of the validation sample.
        val_fraction: Share of each remaining category kept for validation.
    """
    unknown = set(held_out) - {row["category"] for row in queries}
    if unknown:
        raise ValueError(f"held-out categories not in the data: {sorted(unknown)}")
    test = [row["query_id"] for row in queries if row["category"] in held_out]
    remaining = [row for row in queries if row["category"] not in held_out]
    val = _stratified_sample(_ids_by_category(remaining), val_fraction, random.Random(seed))
    train = [row["query_id"] for row in remaining if row["query_id"] not in val]
    return _make_split(
        "unseen_categories",
        f"All photos of the held-out categories {sorted(held_out)} form the test part.",
        seed,
        train,
        val,
        test,
    )


def save_split(split: dict, path: Path) -> Path:
    """Write a split as JSON (one id per line, so that git shows changes clearly)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=1)
        f.write("\n")
    return path


def load_split(path: Path) -> dict:
    """Read a split file."""
    with open(path) as f:
        return json.load(f)


def test_free_query_ids(dataset: str, splits_dir: Path = paths.SPLITS_DIR) -> list[str]:
    """Return the test-free photos of a dataset: the query ids that are in no test part of any split.

    Why: the lighting, the backbone, the feature recipe, the segmenter, and the zero-shot scoring are
    shared by all splits. A selection fits its photos a little, so these settings are selected only
    on photos that no split uses as test photos (see the module description). The photos are in the
    train or val part of every split; they are not a split of their own. Because a zero-shot model
    learns nothing, its training photos are as new to it as its validation photos.

    Args:
        dataset: For example "pix3d"; the split files are read from `<splits_dir>/<dataset>/`.
        splits_dir: The folder of the split files.

    Returns:
        The sorted query ids.

    Raises:
        FileNotFoundError: If a split file of the dataset is missing (the result would include its test photos).
        ValueError: If the splits do not cover the same photos.
    """
    all_photos, test_photos = None, set()
    for name in paths.SPLIT_NAMES:
        split_parts = load_split(paths.split_file(dataset, name, splits_dir))["parts"]
        photos = {query_id for part in PARTS for query_id in split_parts[part]}
        if all_photos is not None and photos != all_photos:
            raise ValueError(f"the splits of {dataset!r} do not cover the same photos ({name!r} differs)")
        all_photos = photos
        test_photos |= set(split_parts["test"])
    return sorted(all_photos - test_photos)
