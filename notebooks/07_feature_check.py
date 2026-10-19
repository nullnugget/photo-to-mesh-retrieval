import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 07: Feature Check

    This notebook checks the feature files that the feature step made (`scripts/05_extract_features.py`):
    the backbone features of the renders and of the query images of one dataset.

    A feature is the vector that the frozen backbone makes from one image (768 numbers for
    DINOv3 ViT-B/16). The step saves two basic parts per image: the class token (`cls`, a summary of
    the whole image) and the masked mean (`masked_mean`, the mean of the patch tokens on the object).
    Every feature recipe is made from these two parts.

    It shows:

    1. optionally, a mini run of the feature step (2 meshes per category, 3 photos of each);
    2. the run report: rows, speed, GPU memory, and repeatability of each feature set;
    3. checks of the files: complete rows, unit-length vectors, no invalid numbers;
    4. a spot check: a few images extracted again directly must give the stored vectors;
    5. a zero-shot sanity check on the tuning photos: recall@1 and recall@5 of each query set and
       each recipe. It uses only the train and val photos of the seen_meshes split; the test photos
       stay unseen. The baselines of the project come from the evaluation step, not from here.

    **How to run**

    - Full files: run `python scripts/05_extract_features.py --dataset pix3d` first. Mini files: press
      the button in section 2 (in the editor), or run the script with `--mini`.
    - Then, from the repository root: `marimo edit notebooks/07_feature_check.py` (interactively), or
      `python notebooks/07_feature_check.py -- --dataset abo` (as a script; it checks the full files if
      they exist, otherwise the mini files; without `--dataset` it checks Pix3D).
    - Section 4 needs the GPU (it loads the backbone). Section 5 only reads the files: a few seconds
      for Pix3D, about a minute for ABO.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules. The cell after it chooses the
    dataset to check (`pix3d` or `abo`); as a script, the dataset comes from `--dataset`.
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image

    # The notebook is in <repository>/notebooks/, so the repository root is one level up.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes the project package "src" importable
    from src import paths  # noqa: E402
    from src.config import load_config  # noqa: E402
    from src.data.catalog import read_table  # noqa: E402
    from src.data.splits import load_split  # noqa: E402
    from src.models.backbones import FEATURE_PARTS, FEATURE_RECIPES, FeatureExtractor, object_masks  # noqa: E402
    from src.models.feature_store import read_features  # noqa: E402
    from src.pipeline import extract_features  # noqa: E402
    from src.retrieval import metrics, search  # noqa: E402

    CFG = load_config()
    BACKGROUND = tuple(CFG["data"]["background_rgb"])
    BACKBONE = CFG["backbone"]["name"]
    SPOT_CHECK_IMAGES = 4  # images per feature set in the spot check
    SPOT_CHECK_SEED = 0
    return (
        BACKBONE,
        BACKGROUND,
        CFG,
        FEATURE_PARTS,
        FEATURE_RECIPES,
        FeatureExtractor,
        Image,
        SPOT_CHECK_IMAGES,
        SPOT_CHECK_SEED,
        extract_features,
        json,
        load_config,
        load_split,
        metrics,
        np,
        object_masks,
        paths,
        read_features,
        read_table,
        search,
    )


@app.cell
def _(mo):
    # In the editor, choose here. As a script: python notebooks/07_feature_check.py -- --dataset abo
    dataset_choice = mo.ui.dropdown(["pix3d", "abo"], value=mo.cli_args().get("dataset") or "pix3d", label="dataset to check")
    dataset_choice
    return (dataset_choice,)


@app.cell
def _(dataset_choice):
    DATASET = dataset_choice.value  # the dataset that every later cell checks
    return (DATASET,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Mini run (optional) and the files to check

    The button runs the feature step on a small sample (`extract_features.run(..., mini=True)`, the
    same function as `scripts/05_extract_features.py --mini`): 2 meshes per category and 3 photos of
    each. It writes only to `test_outputs/extract_features_mini/` and takes about half a minute. Files
    that exist are kept. Then choose which files the rest of the notebook checks: `full`
    (`data/derived/features/`) or `mini`. When the notebook runs as a script, the button is not pressed.
    """)
    return


@app.cell
def _(mo):
    mini_button = mo.ui.run_button(label="Make the mini feature files")
    mini_button
    return (mini_button,)


@app.cell
def _(DATASET, extract_features, load_config, mini_button):
    # Runs only when the button is pressed; otherwise mini_report is None and nothing is extracted.
    mini_report = extract_features.run(DATASET, load_config(), mini=True) if mini_button.value else None
    return (mini_report,)


@app.cell
def _(BACKBONE, DATASET, extract_features, mini_report, mo):
    _ = mini_report  # makes this cell run again after a mini run, so that the new files appear in the list
    ROOTS = {
        name: root
        for name, root in {"full": extract_features.output_root(False), "mini": extract_features.output_root(True)}.items()
        if extract_features.report_file(BACKBONE, DATASET, root).exists()
    }
    mo.stop(not ROOTS, mo.md("No feature files yet. Press the button for a mini run, or run `scripts/05_extract_features.py`."))
    root_choice = mo.ui.dropdown(list(ROOTS), value=next(iter(ROOTS)), label="feature files to check")
    root_choice
    return ROOTS, root_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Run report and file checks

    The next cell reads the report of the step and every feature file of the dataset. One row per
    feature set:

    | Field | Meaning | Expected |
    |:---|:---|:---|
    | `feature_set` | `renders`, `queries_reference` (Pix3D's own masks), or `queries_sam` (SAM masks) | |
    | `rows` | Meshes (renders) or query images (queries) in the file | |
    | `catalog_rows` | Meshes or query photos of the dataset in the catalog (in a mini run, the mini sample) | `rows` + `left_out` |
    | `left_out` | Query photos without an image (no SAM mask, or the crop failed) | 0 |
    | `shape` | Shape of each part: (meshes, views, dim) or (queries, dim) | |
    | `size_mb` | File size | |
    | `images_per_second` | Extraction speed, including reading the files | about 200 or more |
    | `peak_gpu_memory_mb` | Peak GPU memory of PyTorch during the extraction | about 1,000 |
    | `repeat_max_difference` | Largest difference of one number when the first batch is extracted twice | about 0 |
    | `norm_min`, `norm_max` | Smallest and largest vector length over both parts (float16 rounding included) | 1.0 ± 0.002 |
    | `invalid_numbers` | Numbers that are not finite (NaN or infinity) | 0 |
    | `made` | Date and git commit of the file | |
    """)
    return


@app.cell
def _(BACKBONE, DATASET, FEATURE_PARTS, ROOTS, extract_features, json, mo, np, paths, read_features, root_choice):
    root = ROOTS[root_choice.value]
    MINI = root_choice.value == "mini"
    report = json.loads(extract_features.report_file(BACKBONE, DATASET, root).read_text())
    _meshes, _queries = extract_features.catalog_rows(DATASET, MINI)
    TABLES = {}
    _rows = []
    for _set in extract_features.feature_sets(DATASET):
        _file = paths.features_file(BACKBONE, DATASET, _set, root)
        if not _file.exists():
            continue
        TABLES[_set] = read_features(_file)
        _table, _entry = TABLES[_set], report["sets"].get(_set, {})
        _norms = np.concatenate([np.linalg.norm(_table.parts[p].astype(np.float32), axis=-1).ravel() for p in FEATURE_PARTS])
        _rows.append({
            "feature_set": _set,
            "rows": _table.rows,
            "catalog_rows": len(_meshes) if _set == "renders" else len(_queries),
            "left_out": len(_entry.get("left_out", [])),
            "shape": str(_table.parts["cls"].shape),
            "size_mb": _entry.get("size_mb"),
            "images_per_second": _entry.get("images_per_second"),
            "peak_gpu_memory_mb": _entry.get("peak_gpu_memory_mb"),
            "repeat_max_difference": _entry.get("repeat_max_difference"),
            "norm_min": round(float(_norms.min()), 4),
            "norm_max": round(float(_norms.max()), 4),
            "invalid_numbers": int(sum((~np.isfinite(_table.parts[p])).sum() for p in FEATURE_PARTS)),
            "made": f"{_table.info['created']} ({_table.info['git_commit'][:7]}{', changed files' if _table.info['git_dirty'] else ''})",
        })
    print("FILES", _rows)
    _left_out = [f"\n- `{_set}`: `{item['query_id']}` ({item['message']})" for _set, _entry in report["sets"].items() for item in _entry.get("left_out", [])[:10]]
    mo.vstack([mo.ui.table(_rows, selection=None), mo.md("Left-out photos (first 10 per set):" + "".join(_left_out) if _left_out else "No left-out photos.")])
    return MINI, TABLES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Spot check: extract again and compare

    The button loads the backbone and extracts a few random images of each feature set again,
    directly from their image files (`FeatureExtractor.extract_parts`). The result must equal the
    stored vectors, up to the float16 rounding of the store (about 0.001). A larger difference means
    that a row of the file belongs to another image (a wrong order). As a script, the check runs
    without the button.

    | Field | Meaning | Expected |
    |:---|:---|:---|
    | `feature_set` | The feature set | |
    | `images` | Images checked | 4 |
    | `max_difference` | Largest difference of one number between the stored and the new vector | below 0.002 |
    """)
    return


@app.cell
def _(mo):
    spot_button = mo.ui.run_button(label="Run the spot check (GPU)")
    spot_button
    return (spot_button,)


@app.cell
def _(BACKGROUND, CFG, DATASET, FEATURE_PARTS, FeatureExtractor, Image, MINI, SPOT_CHECK_IMAGES, SPOT_CHECK_SEED, TABLES, extract_features, mo, np, paths, read_table, spot_button):
    mo.stop(not spot_button.value and mo.app_meta().mode != "script", mo.md("Press the button to run section 4."))
    _extractor = FeatureExtractor(CFG["backbone"])
    _rng = np.random.default_rng(SPOT_CHECK_SEED)
    _render_dirs = {m["mesh_id"]: m["render_dir"] for m in read_table(paths.MESHES_CSV)}
    _query_rows = {q["query_id"]: q for q in read_table(paths.QUERIES_CSV)}
    _rows = []
    for _set, _table in TABLES.items():
        _picks = _rng.choice(_table.rows, size=min(SPOT_CHECK_IMAGES, _table.rows), replace=False)
        _worst = 0.0
        for _row in _picks:
            if _set == "renders":
                _view = int(_rng.integers(_table.parts["cls"].shape[1]))
                _file, _stored = paths.resolve(_render_dirs[_table.ids[_row]]) / f"view_{_view:02d}.png", {p: _table.parts[p][_row, _view] for p in FEATURE_PARTS}
            else:
                _query = _query_rows[_table.ids[_row]]
                _sam = extract_features.QUERY_SETS[DATASET][_set] == extract_features.SAM_IMAGES
                _file = extract_features.sam_query_file(extract_features.sam_queries_root(MINI), _query) if _sam else paths.resolve(_query["query_path"])
                _stored = {p: _table.parts[p][_row] for p in FEATURE_PARTS}
            _image = np.asarray(Image.open(_file).convert("RGB"))[None]
            _new = _extractor.extract_parts(_image, object_masks(_image, BACKGROUND))
            _worst = max(_worst, max(float(np.abs(_new[p][0] - _stored[p].astype(np.float32)).max()) for p in FEATURE_PARTS))
        _rows.append({"feature_set": _set, "images": len(_picks), "max_difference": round(_worst, 5)})
    print("SPOT_CHECK", _rows)
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Zero-shot sanity check on the tuning photos

    The next cell ranks all meshes of the file for each tuning photo (the train and val photos of the
    seen_meshes split) with the view score: the cosine similarity to the best-matching view of each
    mesh. It does this for every query set and every recipe. No test photo is used.

    For Pix3D with the full files, `queries_reference` with the `cls` recipe repeats the zero-shot
    result of the segment check (notebook 05, the same photos, renders, and recipe): recall@1 about
    67.7 %, and `queries_sam` about 59.5 %. A large difference means that the stored features differ
    from the earlier ones. With the mini files the gallery is very small, so the values are high and
    only show that the files work.

    | Field | Meaning |
    |:---|:---|
    | `query_set` | The query set |
    | `recipe` | `cls`, `masked_mean`, or `cls_masked_mean` (both parts, joined) |
    | `photos` | Tuning photos in the query set |
    | `gallery_meshes` | Meshes ranked for each photo |
    | `R@1`, `R@5` | Share of photos whose correct mesh is first, or in the first 5 (%) |
    | `R@1_interval` | 95 % bootstrap interval of R@1 |
    """)
    return


@app.cell
def _(DATASET, FEATURE_RECIPES, TABLES, load_split, metrics, mo, np, paths, search):
    mo.stop("renders" not in TABLES, mo.md("No render file: section 5 needs it."))
    _renders = TABLES["renders"]
    _column = {mesh_id: index for index, mesh_id in enumerate(_renders.ids)}
    _split = load_split(paths.split_file(DATASET, "seen_meshes"))["parts"]
    _tuning = set(_split["train"]) | set(_split["val"])
    _rows = []
    for _set, _table in TABLES.items():
        if _set == "renders":
            continue
        _queries = _table.select([query_id for query_id in _table.ids if query_id in _tuning])
        _correct = np.array([_column[mesh_id] for mesh_id in _queries.labels["mesh_id"]])
        for _recipe in FEATURE_RECIPES:
            _ranks = metrics.correct_ranks(search.view_scores(_queries.vectors(_recipe), _renders.vectors(_recipe)), _correct)
            _low, _high = metrics.bootstrap_interval(_ranks <= 1)
            _rows.append({
                "query_set": _set,
                "recipe": _recipe,
                "photos": _queries.rows,
                "gallery_meshes": _renders.rows,
                "R@1": round(100 * metrics.recall_at_k(_ranks, 1), 1),
                "R@1_interval": f"{100 * _low:.1f} to {100 * _high:.1f}",
                "R@5": round(100 * metrics.recall_at_k(_ranks, 5), 1),
            })
    print("ZERO_SHOT_TUNING", _rows)
    mo.ui.table(_rows, selection=None)
    return


if __name__ == "__main__":
    app.run()
