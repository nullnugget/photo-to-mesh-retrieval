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

Cross-validation by mesh (folds of "unseen_meshes"). The val part of "unseen_meshes" holds only a
tenth of the meshes (Pix3D: 39). Photos of one mesh succeed or fail together, so a val result on 39
meshes depends mostly on which meshes were drawn, and it can mislead (Pix3D: val +6 for a setting
that gave 0 on the test part). A cross-validation uses every non-test mesh for validation once:
the non-test meshes are divided into k folds (stratified by category); fold k is a split of its own
("unseen_meshes_fold<k>") whose val part is the photos of the fold's meshes, whose train part is
the photos of all other non-test meshes, and whose test part is the unchanged test part. k
trainings, one per fold, together validate on every non-test mesh (Pix3D: 315 meshes, 8,020
photos); `src/train/cross_validation.py` pools their results. The test part stays as it is.

Cross-validation by category (folds of "unseen_categories"). The val part of "unseen_categories" is
a photo sample of the training categories: nearly all its meshes have training photos (Pix3D: 243 of
244), so it measures seen meshes, not new categories. A leave-one-category-out cross-validation
uses each non-test category once as the val part ("unseen_categories_fold<k>": val = all photos of
category k, train = the photos of the other non-test categories, test = the unchanged test part),
so every non-test photo is validated by a model that never saw its category.

Joint training of several datasets by category. The folds of the first dataset (Pix3D) decide the
folds: another dataset (ABO) gets one fold per category of the first one, with the same order and
numbers. Its val part is its photos of that category (empty if it has none, for example ABO has no
"misc"), and its other non-test photos are its train part. Then the model of fold k has seen the
held-out category in no dataset.

Near relatives ("relatives" in `data.splits.unseen_categories`). A category that is nearly the same
object as a held-out one (ABO shelf for Pix3D bookcase, ABO stool for chair) also teaches the
held-out category. So every unseen_categories training leaves out the photos of the near relatives
of its held-out categories, in every dataset (`left_out_categories`); the held-out categories are
the test categories of each dataset and, in a fold, the category of the fold:

    test part:         bookcase, wardrobe (Pix3D)  →  also out: shelf, tv_stand, cabinet, dresser (ABO)
    fold "chair":      chair (Pix3D and ABO)       →  also out: stool, bench, bean_bag (ABO)

The split files stay unchanged; the training reads the rule from the settings. A training of Pix3D
alone loses nothing: no Pix3D category is a near relative of another one in the settings.

Every split is repeatable: it depends only on the input rows and a seed. Split files are small
JSON files in `data/splits/<dataset>/` and are tracked in git, so that every run uses the same
split. A split works for any dataset: it needs only the query id, the category, and the mesh id
of each photo.
"""

import json
import random
from collections import Counter, defaultdict
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


def fold_name(split_name: str, fold: int) -> str:
    """Return the name of one fold of a split, for example "unseen_meshes_fold1"."""
    return f"{split_name}_fold{fold}"


def mesh_folds(base: dict, queries: list[dict], folds: int, seed: int = 42) -> list[dict]:
    """Divide the non-test meshes of a split into folds; return one split per fold.

    Args:
        base: The split whose non-test meshes are divided (normally "unseen_meshes"). Its test part
            stays the test part of every fold.
        queries: Catalog rows (query_id, category, mesh_id) of every photo of the split.
        folds: Number of folds k.
        seed: Seed of the random order of the meshes.

    In each category, the meshes go one by one to the fold that has the fewest meshes of that
    category (ties: the fewest photos, then the lowest fold number), largest meshes (most photos)
    first. So every fold gets about the same number of meshes of each category (a category with
    at least k meshes is in every fold), and the photo counts stay as even as the meshes allow.

    Raises:
        ValueError: If a mesh has photos in the test part and outside it, or if the rows do not
            cover the photos of the split.
    """
    test = set(base["parts"]["test"])
    non_test = set(base["parts"]["train"]) | set(base["parts"]["val"])
    if {row["query_id"] for row in queries} != non_test | test:
        raise ValueError(f"the rows do not cover exactly the photos of the split {base['name']!r}")
    rows = [row for row in queries if row["query_id"] not in test]
    shared = {row["mesh_id"] for row in rows} & {row["mesh_id"] for row in queries if row["query_id"] in test}
    if shared:
        raise ValueError(f"{len(shared)} meshes have photos in the test part and outside it; folds by mesh need a split by mesh")
    rng = random.Random(seed)
    fold_of_mesh: dict[str, int] = {}
    for _category, group in _group_by(rows, "category").items():
        photos = Counter(row["mesh_id"] for row in group)
        meshes = sorted(photos)
        rng.shuffle(meshes)
        meshes.sort(key=lambda mesh: -photos[mesh])  # stable sort: equal counts keep the random order
        mesh_count, photo_count = [0] * folds, [0] * folds
        for mesh in meshes:
            k = min(range(folds), key=lambda i: (mesh_count[i], photo_count[i], i))
            fold_of_mesh[mesh] = k
            mesh_count[k] += 1
            photo_count[k] += photos[mesh]
    made = []
    for k in range(folds):
        val = [row["query_id"] for row in rows if fold_of_mesh[row["mesh_id"]] == k]
        train = [row["query_id"] for row in rows if fold_of_mesh[row["mesh_id"]] != k]
        split = _make_split(
            fold_name(base["name"], k + 1),
            f"Fold {k + 1} of {folds} of the non-test meshes of {base['name']!r} (by mesh, stratified by category): "
            f"val = the photos of the fold's meshes, train = the photos of the other non-test meshes, test = the test part of {base['name']!r}.",
            seed, train, val, base["parts"]["test"],
        )
        split.update({"base_split": base["name"], "fold": k + 1, "folds": folds})
        made.append(split)
    return made


def category_folds(base: dict, queries: list[dict], seed: int = 42, categories: list[str] | None = None) -> list[dict]:
    """One fold per non-test category of a split (leave-one-category-out); return one split per fold.

    Args:
        base: The split whose non-test categories become folds (normally "unseen_categories"). Its
            test part stays the test part of every fold.
        queries: Catalog rows (query_id, category, mesh_id) of every photo of the split.
        seed: Kept in the split record (the folds themselves are not random).
        categories: The categories of the folds, in fold order (None = the non-test categories of
            this split, sorted). A joint training gives the categories of its first dataset; a
            category without photos here gets an empty val part.

    Fold k holds the k-th category.

    Raises:
        ValueError: If a category has photos in the test part and outside it, or if the rows do not
            cover the photos of the split.
    """
    test = set(base["parts"]["test"])
    non_test = set(base["parts"]["train"]) | set(base["parts"]["val"])
    if {row["query_id"] for row in queries} != non_test | test:
        raise ValueError(f"the rows do not cover exactly the photos of the split {base['name']!r}")
    rows = [row for row in queries if row["query_id"] not in test]
    shared = {row["category"] for row in rows} & {row["category"] for row in queries if row["query_id"] in test}
    if shared:
        raise ValueError(f"the categories {sorted(shared)} have photos in the test part and outside it; folds by category need a split by category")
    own = sorted({row["category"] for row in rows})
    led = categories is not None
    categories = list(categories) if led else own
    made = []
    for k, category in enumerate(categories):
        val = [row["query_id"] for row in rows if row["category"] == category]
        train = [row["query_id"] for row in rows if row["category"] != category]
        split = _make_split(
            fold_name(base["name"], k + 1),
            f"Fold {k + 1} of {len(categories)} of {base['name']!r} (leave one category out): val = every photo of the category "
            f"{category!r}, train = the photos of the other non-test categories, test = the test part of {base['name']!r}."
            + (" The folds follow the categories of the first dataset of a joint training." if led and categories != own else ""),
            seed, train, val, base["parts"]["test"],
        )
        split.update({"base_split": base["name"], "fold": k + 1, "folds": len(categories), "held_out_category": category})
        made.append(split)
    return made


def carry_over_parts(old: dict, fresh: dict, queries: list[dict], by_mesh: bool,
                     photo_key=lambda query_id: query_id.rsplit("/", 1)[-1]) -> dict:
    """Keep the parts of an earlier split after a catalog change (for example merged gallery entries).

    Why: the shared settings were chosen only on photos outside every test part (the test-free
    photos). A photo that helped to choose a setting must never move into a test part, or that test
    part is no longer clean. So each photo keeps its earlier part, and a new test part is a subset
    of the earlier one.

    Args:
        old: The earlier split (its parts decide).
        fresh: The split made from the new catalog (it gives the name, the description, the seed, and
            the part of a photo that the earlier split does not have).
        queries: The new catalog rows (query_id, mesh_id, category) of the split's photos.
        by_mesh: True for a split by mesh ("unseen_meshes"): all photos of a mesh must be in one part.
            A mesh whose photos are in several parts (two entries that were merged) goes as a whole
            into the non-test part with the most of its photos (a tie: train). Its test photos become
            training photos; no photo moves into the test part.
        photo_key: Identifies a photo across the change (default: the last part of the query id, the
            image id; the category part of an ABO query id can change).

    Raises:
        ValueError: If photo keys repeat.
    """
    keys = [photo_key(row["query_id"]) for row in queries]
    if len(set(keys)) != len(keys):
        raise ValueError("photo keys repeat; give a photo_key that is unique within the dataset")
    old_part = {photo_key(q): part for part, ids in old["parts"].items() for q in ids}
    fresh_part = {q: part for part, ids in fresh["parts"].items() for q in ids}
    part = {row["query_id"]: old_part.get(photo_key(row["query_id"]), fresh_part[row["query_id"]]) for row in queries}
    if by_mesh:
        by = defaultdict(Counter)
        for row in queries:
            by[row["mesh_id"]][part[row["query_id"]]] += 1
        target = {}
        for mesh_id, counts in by.items():
            if len(counts) > 1:
                non_test = {p: n for p, n in counts.items() if p != "test"}
                target[mesh_id] = max(non_test, key=lambda p: (non_test[p], p == "train"))
        for row in queries:
            if row["mesh_id"] in target:
                part[row["query_id"]] = target[row["mesh_id"]]
    lists = {p: [q for q, x in part.items() if x == p] for p in PARTS}
    made = _make_split(fresh["name"], fresh["description"] + " Parts carried over from the earlier split (a catalog "
                       "change); a mesh in several parts went whole into a non-test part.", fresh["seed"],
                       lists["train"], lists["val"], lists["test"])
    made.update({k: v for k, v in fresh.items() if k not in made})
    return made


def left_out_categories(held_out: set[str], relatives: dict[str, list[str]] | None) -> set[str]:
    """Return the categories that an unseen_categories training leaves out: the held-out ones and their near relatives.

    Args:
        held_out: The held-out categories of the training (the test categories of each dataset and,
            in a fold, the category of the fold).
        relatives: `data.splits.unseen_categories.relatives`: category -> its near relatives.
    """
    relatives = relatives or {}
    return set(held_out) | {r for category in held_out for r in relatives.get(category, [])}


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
