"""The evaluation step: retrieval metrics of one dataset for a group of photos, recorded as a run.

Zero-shot mode (the only mode until the embedding heads exist): the stored backbone features are
compared directly (`src/models/feature_store.py`), without any training. One call evaluates every
combination of the chosen query sets, feature recipes, and scorings on the same photos.

Photo groups (`--photos`):

| Group     | Photos                                                   | Use                                    |
|-----------|----------------------------------------------------------|----------------------------------------|
| test_free | the photos in no test part of any split (`--split` not used) | selecting the shared settings      |
| train     | the train part of `--split`                              | checks                                 |
| val       | the val part of `--split`                                | progress; selections of that split     |
| test      | the test part of `--split`; needs `allow_test`           | the reported number, once              |

Why the test-free photos: the lighting, the backbone, the feature recipe, the segmenter, and the
zero-shot scoring are shared by all three splits (one set of each). A selection fits the photos
that it is made on a little (the best of many settings is partly lucky on exactly these photos).
So these settings are selected only on photos that no split uses as test photos
(`src/data/splits.py`, `test_free_query_ids`). In zero-shot mode nothing is trained, so the
training photos among them are as new to the model as validation photos.

Why the test lock: a test part gives the reported number of its split. It is evaluated once,
after all selections are fixed. `allow_test=True` (`--allow-test`) makes this a deliberate act.

The gallery is every mesh of the renders file. Query photos that a query set does not have (for
example, a photo without a SAM mask) are left out of that set and counted in the results.

Outputs of one call (a mini run writes into `test_outputs/evaluate_mini/` instead):
    runs/<run_name>/config.yaml                                 the settings of the call
    runs/<run_name>/metrics.json                                every result, with the per-category and subset values
    runs/<run_name>/per_query/<query_set>_<recipe>_<scoring>.csv  one row per photo: ranks and first result
    runs/registry.csv                                           one row per result
"""

import csv
import datetime
import json
from pathlib import Path

import numpy as np

from src import paths, runs
from src.config import save_config
from src.data.catalog import read_table
from src.data.splits import load_split, test_free_query_ids
from src.models.backbones import FEATURE_RECIPES
from src.models.feature_store import FeatureTable, read_features
from src.pipeline import extract_features
from src.retrieval.evaluate import SCORINGS, Gallery, Queries, evaluate

TEST_FREE = "test_free"
PHOTO_GROUPS = (TEST_FREE, "train", "val", "test")
MODE = "zero_shot"
PER_QUERY_COLUMNS = ["query_id", "category", "mesh_id", "rank", "rank_inside_category", "first_mesh_id", "first_score", "correct_score"]


class TestPhotosLocked(RuntimeError):
    """A test part was asked for without allow_test."""


def output_dirs(mini: bool) -> dict[str, Path]:
    """Return the run folder root and the registry: the real ones, or test copies for a mini run."""
    if mini:
        base = paths.TEST_OUTPUTS_DIR / "evaluate_mini"
        return {"runs": base / "runs", "registry": base / "registry.csv"}
    return {"runs": paths.RUNS_DIR, "registry": paths.REGISTRY_CSV}


def photo_ids(dataset: str, photos: str, split: str | None) -> list[str]:
    """Return the query ids of a photo group (see the table in the module description)."""
    if photos == TEST_FREE:
        if split:
            raise ValueError("the test-free photos belong to no split; leave the split out")
        return test_free_query_ids(dataset)
    if photos not in PHOTO_GROUPS:
        raise ValueError(f"photos must be one of {PHOTO_GROUPS}, got {photos!r}")
    if not split:
        raise ValueError(f"the photo group {photos!r} needs a split")
    return list(load_split(paths.split_file(dataset, split))["parts"][photos])


def _queries(table: FeatureTable, recipe: str) -> Queries:
    return Queries(table.vectors(recipe), table.labels["query_id"], table.labels["mesh_id"], table.labels["category"])


def _gallery(table: FeatureTable, recipe: str) -> Gallery:
    return Gallery(table.vectors(recipe), table.labels["mesh_id"], table.labels["category"])


def _write_per_query(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_QUERY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(dataset: str, cfg: dict, photos: str = TEST_FREE, split: str | None = None,
        query_sets: list[str] | None = None, recipes: list[str] | None = None, scorings: list[str] | None = None,
        alpha: float | None = None, backbone: str | None = None, allow_test: bool = False,
        tag: str | None = None, mini: bool = False) -> dict:
    """Evaluate every combination of query set, recipe, and scoring on one photo group; record a run.

    Args:
        dataset: "pix3d" or "abo".
        cfg: The merged settings (saved into the run folder).
        photos: One of PHOTO_GROUPS.
        split: The split of the photo group; None for the test-free photos.
        query_sets: Query feature sets (None = every query set of the dataset).
        recipes: Feature recipes (None = `backbone.feature_recipe`).
        scorings: Scorings from SCORINGS (None = `backbone.zero_shot.scoring`).
        alpha: Weight of the view score in the combined score (None = `backbone.zero_shot.alpha`).
        backbone: Backbone of the feature files (None = `backbone.name`).
        allow_test: Must be True to evaluate a test part (see the module description).
        tag: Extra word for the run name.
        mini: True = the mini feature files of the feature step, outputs in test_outputs/evaluate_mini/.

    Returns:
        The content of metrics.json.

    Raises:
        TestPhotosLocked: If photos is "test" and allow_test is False.
    """
    if photos == "test" and not allow_test:
        raise TestPhotosLocked(
            "a test part gives the reported number of its split and is evaluated once, after all selections "
            "are fixed. Select settings on the test-free photos; use allow_test (--allow-test) for the final evaluation."
        )
    backbone = backbone or cfg["backbone"]["name"]
    query_sets = query_sets or list(extract_features.QUERY_SETS[dataset])
    recipes = recipes or [cfg["backbone"]["feature_recipe"]]
    zero_shot = cfg["backbone"]["zero_shot"]
    scorings = scorings or [zero_shot["scoring"]]
    alpha = zero_shot["alpha"] if alpha is None else alpha
    for name, values, allowed in (("query set", query_sets, extract_features.QUERY_SETS[dataset]),
                                  ("recipe", recipes, FEATURE_RECIPES), ("scoring", scorings, SCORINGS)):
        unknown = [v for v in values if v not in allowed]
        if unknown:
            raise ValueError(f"unknown {name} {unknown}; allowed: {list(allowed)}")

    features_root, out = extract_features.output_root(mini), output_dirs(mini)
    wanted = photo_ids(dataset, photos, split)
    renders_file = paths.features_file(backbone, dataset, "renders", features_root)
    renders = read_features(renders_file)
    flags = {q["query_id"]: q for q in read_table(paths.QUERIES_CSV) if q["dataset"] == dataset}
    used_files = {str(renders_file.relative_to(paths.ROOT)): runs.file_sha256(renders_file)}

    results, per_query = [], {}
    for query_set in query_sets:
        query_file = paths.features_file(backbone, dataset, query_set, features_root)
        table = read_features(query_file)
        used_files[str(query_file.relative_to(paths.ROOT))] = runs.file_sha256(query_file)
        present = set(table.ids.tolist())
        selected = table.select([q for q in wanted if q in present])
        not_in_set = len(wanted) - selected.rows  # photos of the group without an image in this set
        for recipe in recipes:
            queries, gallery = _queries(selected, recipe), _gallery(renders, recipe)
            for scoring in scorings:
                result, rows = evaluate(queries, gallery, scoring, alpha if scoring == "combined" else None, flags)
                key = f"{query_set}_{recipe}_{scoring}"
                per_query[key] = rows
                results.append({"query_set": query_set, "recipe": recipe, "scoring": scoring,
                                "alpha": alpha if scoring == "combined" else None,
                                "photos_not_in_query_set": not_in_set, **result})
                print(f"{query_set:18s} {recipe:16s} {scoring:9s} photos {result['photos']:6d}  "
                      f"R@1 {100 * result['R@1']:5.1f} ({100 * result['R@1_low']:.1f} to {100 * result['R@1_high']:.1f}; "
                      f"meshes {100 * result['R@1_low_meshes']:.1f} to {100 * result['R@1_high_meshes']:.1f})  "
                      f"R@5 {100 * result['R@5']:5.1f}  R@10 {100 * result['R@10']:5.1f}  "
                      f"Cat@1 {100 * result['cat@1']:5.1f}  macro R@1 {100 * result['macro_R@1']:5.1f}")

    # Record the run: folder, metrics, per-query rows, and one registry row per result.
    run_tag = "_".join(part for part in (MODE, photos, split, tag) if part)
    run_name = runs.make_run_name(dataset, backbone, run_tag, runs_dir=out["runs"])
    run_dir = paths.ensure_dir(out["runs"] / run_name)
    save_config(cfg, run_dir / "config.yaml")
    for key, rows in per_query.items():
        _write_per_query(rows, run_dir / "per_query" / f"{key}.csv")
    commit, dirty = runs.git_state()
    renders_info = renders.info
    record = {
        "run_name": run_name,
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "mode": MODE,
        "dataset": dataset,
        "photos": photos,
        "split": split,
        "photo_count": len(wanted),
        "backbone": backbone,
        "feature_files": used_files,  # path -> SHA-256 of the file content
        "lighting": renders_info.get("lighting"),
        "rig": renders_info.get("rig"),
        "mini": mini,
        "results": results,
    }
    (run_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    for result in results:
        runs.append_registry({
            "run_name": run_name,
            "date": record["date"][:10],
            "git_commit": commit[:7],
            "git_dirty": dirty,
            "mode": MODE,
            "datasets": dataset,
            "split": split or "",
            "photos": photos,
            "photo_count": result["photos"],
            "query_set": result["query_set"],
            "backbone": backbone,
            "recipe": result["recipe"],
            "scoring": result["scoring"],
            "alpha": result["alpha"] if result["alpha"] is not None else "",
            "render_config": renders_info.get("rig", ""),
            "lighting": renders_info.get("lighting", ""),
            "R@1": round(100 * result["R@1"], 2),
            "R@5": round(100 * result["R@5"], 2),
            "R@10": round(100 * result["R@10"], 2),
            "cat@1": round(100 * result["cat@1"], 2),
            "macro_R@1": round(100 * result["macro_R@1"], 2),
            "notes": "mini" if mini else "",
        }, registry=out["registry"])
    print(f"run -> {run_dir.relative_to(paths.ROOT)}; {len(results)} rows -> {out['registry'].relative_to(paths.ROOT)}")
    return record
