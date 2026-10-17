"""Data splits: which query photos are used for training, for validation, and for the test.

A split divides the query photos into three parts:

- train: photos that the model learns from;
- val:   photos that select checkpoints and settings during development;
- test:  photos that give the reported results. They are never used to choose anything.

The gallery (the 3D shapes that a search can return) always contains every shape, also the
shapes whose photos are all in the test part. That is the real use case: a new 3D shape enters
the database without any photos.

Three splits answer three different questions:

1. Image-level split ("pix3d_image_split"): can the model find a known shape from a new photo?
   Photos are split at random; most shapes have photos in every part. The test part is exactly
   the 15 % hold-out of v1 (same seed, same shuffle), so v1 and v2 results are comparable.
2. Unseen-shape split ("unseen_shapes"): can the model find a shape that it never saw in a photo?
   Shapes are split; all photos of a shape go to the same part.
3. Leave-categories-out split ("leave_categories_out"): does the model work for a whole category
   that it never trained on? All photos of the held-out categories form the test part.

Every split is repeatable: it depends only on the input lists and a seed. Split files are small
JSON files in `data/splits/` and are tracked in git, so that every run uses the same split.
"""

import json
import random
from collections import defaultdict
from pathlib import Path


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


def image_level_split(
    annotations_query_ids: list[str],
    queries: list[dict],
    seed: int = 42,
    v1_holdout_fraction: float = 0.15,
    val_fraction_of_train: float = 0.10,
) -> dict:
    """Split photos at random, with the same test part as v1.

    Args:
        annotations_query_ids: The query ids of ALL Pix3D photos, in the order of pix3d.json.
            The v1 split shuffled exactly this list, so the order matters.
        queries: The catalog rows of the photos that exist as queries (query_id, category).
            Photos that could not be prepared are not in this list; they leave the split.
        seed: Seed of the v1 shuffle (42 in v1) and of the validation sample.
        v1_holdout_fraction: The v1 hold-out size (0.15): the last 15 % after the shuffle.
        val_fraction_of_train: Part of each category's training photos kept apart for validation.

    The v1 code was: `random.seed(42); random.shuffle(entries); cut = int(len(entries) * 0.85)`.
    `random.Random(42).shuffle` gives the same order as that code.
    """
    shuffled = list(annotations_query_ids)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - v1_holdout_fraction))
    available = {q["query_id"]: q for q in queries}
    train_pool = [qid for qid in shuffled[:cut] if qid in available]
    test = [qid for qid in shuffled[cut:] if qid in available]

    # Validation: a stratified sample of the training photos.
    by_category: dict[str, list[str]] = defaultdict(list)
    for qid in train_pool:
        by_category[available[qid]["category"]].append(qid)
    val = _stratified_sample(dict(sorted(by_category.items())), val_fraction_of_train, random.Random(seed))
    train = [qid for qid in train_pool if qid not in val]
    return _make_split(
        "pix3d_image_split",
        "Photos split at random. The test part is the v1 hold-out (seed 42, last 15 % after the shuffle).",
        seed,
        train,
        val,
        test,
    )


def unseen_shape_split(queries: list[dict], seed: int = 42, fractions: dict | None = None) -> dict:
    """Split the shapes of each category into train, val, and test; each photo follows its shape.

    Args:
        queries: Catalog rows (query_id, category, shape_id).
        seed: Seed of the random shape assignment.
        fractions: Shares of shapes, for example {"train": 0.7, "val": 0.1, "test": 0.2}.

    In every category with at least 3 shapes, val and test get at least 1 shape each.
    """
    fractions = fractions or {"train": 0.70, "val": 0.10, "test": 0.20}
    rng = random.Random(seed)
    part_of_shape: dict[str, str] = {}
    for category, rows in _group_by(queries, "category").items():
        shapes = sorted({row["shape_id"] for row in rows})
        rng.shuffle(shapes)
        n = len(shapes)
        n_test = max(1, round(n * fractions["test"])) if n >= 3 else 0
        n_val = max(1, round(n * fractions["val"])) if n >= 3 else 0
        for i, shape in enumerate(shapes):
            part_of_shape[shape] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    parts = {"train": [], "val": [], "test": []}
    for row in queries:
        parts[part_of_shape[row["shape_id"]]].append(row["query_id"])
    return _make_split(
        "unseen_shapes",
        "Shapes split within each category (about 70/10/20); all photos of a shape are in one part.",
        seed,
        parts["train"],
        parts["val"],
        parts["test"],
    )


def leave_categories_out_split(
    queries: list[dict], held_out: list[str], seed: int = 42, val_fraction: float = 0.10
) -> dict:
    """Use whole categories as the test part.

    Args:
        queries: Catalog rows (query_id, category).
        held_out: Categories whose photos form the test part, for example ["bookcase", "wardrobe"].
        seed: Seed of the validation sample.
        val_fraction: Part of each remaining category kept apart for validation.
    """
    unknown = set(held_out) - {row["category"] for row in queries}
    if unknown:
        raise ValueError(f"held-out categories not in the data: {sorted(unknown)}")
    test = [row["query_id"] for row in queries if row["category"] in held_out]
    remaining = [row for row in queries if row["category"] not in held_out]
    by_category = {c: [r["query_id"] for r in rows] for c, rows in _group_by(remaining, "category").items()}
    val = _stratified_sample(by_category, val_fraction, random.Random(seed))
    train = [row["query_id"] for row in remaining if row["query_id"] not in val]
    return _make_split(
        "leave_categories_out",
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
