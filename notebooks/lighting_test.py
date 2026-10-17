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
    # Lighting test

    **Question:** which light setup makes the grey renders look most like the photos, to the
    backbone? The renders are the only view of a mesh that the model gets, and in a grey render
    the shading is the only cue to the 3D form. The light setup decides that shading.

    **Method** (no training needed):

    1. Take a small sample of Pix3D meshes: 3 per category, 27 in total.
    2. Render each mesh from all 25 cameras with each light setup L0 to L4 (`configs/render.yaml`).
    3. Turn every render and every query photo of these meshes into a DINOv3 feature vector.
    4. For each light setup, search the 27 meshes with every query photo and measure how often the
       correct mesh comes first (R@1). The frozen backbone already places photos and renders in one
       feature space, so this "zero-shot" search works before any training.
    5. Compare the light setups with confidence intervals, and look at the renders by eye.

    | Setup | Light |
    |:---|:---|
    | L0 | One point light at the camera (the v1 setup) |
    | L1 | The same position, with less ambient light and smaller highlights |
    | L2 | A key light 35° above and 35° left of the viewing direction (moves with the camera) |
    | L3 | L2 plus a weaker fill light from the right |
    | L4 | One fixed light from straight above, like a ceiling light |

    **Queries:** the train and val photos of the sample meshes (no training happens here, so
    train photos are not "seen"). The test photos stay out, so that the choice of the light does
    not use the test set.

    **Limits:** the gallery has only 27 meshes, so the task is easier than the real one (R@10 is
    close to 100 % for every setup; R@1 and Cat@1 carry the comparison). The ranking is for the
    frozen backbone; after training the heads, it could change, so a short training run checks
    the best one or two setups later.

    **Outputs:**

    - renders: `data/derived/renders_lighting_test/<setup>/<mesh id>/view_00.png` ... `view_24.png`
      (temporary; deleted after the decision);
    - results: `runs/<date>_pix3d_<backbone>_lighting_test/` (settings, metrics, contact sheets),
      written by the last section.

    **How to run:** `marimo edit notebooks/lighting_test.py`, then press the buttons in sections 3
    and 9. As a script (`python notebooks/lighting_test.py`), all steps run and the results are saved.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules, and sets the experiment settings:

    | Setting | Value | Meaning |
    |:---|:---|:---|
    | `MESHES_PER_CATEGORY` | 3 | Meshes sampled from each of the 9 Pix3D categories |
    | `MIN_PHOTOS` | 3 | A mesh needs at least this many non-test photos to be sampled |
    | `MAX_PHOTOS_PER_MESH` | 30 | At most this many photos per mesh are used, so that a few very popular meshes do not dominate |
    | `SEED` | 0 | Fixed seed of the random sample |
    | `LIGHTS` | L0 ... L4 | The light setups to compare (all presets in `configs/render.yaml`) |
    | `ALPHA` | from `configs/train.yaml` | Weight of the view score in the combined score |
    """)
    return


@app.cell
def _():
    import datetime
    import json
    import random
    import sys
    import time
    from collections import Counter
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    # The notebook is in <repository>/notebooks/, so the repository root is one level up.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes the project package "src" importable
    from src import paths  # noqa: E402
    from src.config import load_config, save_config  # noqa: E402
    from src.data.catalog import read_table  # noqa: E402
    from src.data.splits import load_split  # noqa: E402
    from src.models.backbones import FeatureExtractor, object_masks  # noqa: E402
    from src.render.renderer import MeshRenderer  # noqa: E402
    from src.retrieval import metrics, search  # noqa: E402
    from src.runs import git_state, make_run_name  # noqa: E402

    CFG = load_config()
    MESHES_PER_CATEGORY = 3
    MIN_PHOTOS = 3
    MAX_PHOTOS_PER_MESH = 30
    SEED = 0
    LIGHTS = list(CFG["render"]["lighting_presets"])
    ALPHA = CFG["train"]["scoring"]["alpha"]
    BACKGROUND = tuple(CFG["data"]["background_rgb"])
    RENDER_DIR = paths.LIGHTING_TEST_DIR
    VIEWS = 25
    return (
        ALPHA,
        BACKGROUND,
        CFG,
        Counter,
        FeatureExtractor,
        Image,
        ImageDraw,
        LIGHTS,
        MAX_PHOTOS_PER_MESH,
        MESHES_PER_CATEGORY,
        MIN_PHOTOS,
        MeshRenderer,
        RENDER_DIR,
        SEED,
        VIEWS,
        datetime,
        git_state,
        json,
        load_split,
        make_run_name,
        metrics,
        np,
        object_masks,
        paths,
        plt,
        random,
        read_table,
        save_config,
        search,
        time,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Sample of meshes and query photos

    The next cell samples the meshes and their photos:

    1. It reads the Pix3D meshes and queries from the catalog, and the test part of the
       `seen_meshes` split (these photos are never used here).
    2. For each category, it takes 3 random meshes among those with at least 3 non-test photos.
    3. For each sampled mesh, it takes its non-test photos (at most 30, at random).

    | Field | Meaning |
    |:---|:---|
    | `category` | Pix3D category |
    | `mesh_id` | The sampled mesh |
    | `photos_available` | Its train and val photos |
    | `photos_used` | The photos used as queries (at most 30) |
    """)
    return


@app.cell
def _(
    MAX_PHOTOS_PER_MESH,
    MESHES_PER_CATEGORY,
    MIN_PHOTOS,
    SEED,
    load_split,
    mo,
    paths,
    random,
    read_table,
):
    _test_ids = set(load_split(paths.split_file("pix3d", "seen_meshes"))["parts"]["test"])
    _queries = [q for q in read_table(paths.QUERIES_CSV) if q["dataset"] == "pix3d" and q["query_id"] not in _test_ids]
    _photos_of = {}
    for _q in _queries:
        _photos_of.setdefault(_q["mesh_id"], []).append(_q)
    _meshes = [m for m in read_table(paths.MESHES_CSV) if m["dataset"] == "pix3d"]

    _rng = random.Random(SEED)
    sample_meshes, sample_queries, _rows = [], [], []
    for _category in sorted({m["category"] for m in _meshes}):
        _pool = sorted(
            (m for m in _meshes if m["category"] == _category and len(_photos_of.get(m["mesh_id"], [])) >= MIN_PHOTOS),
            key=lambda m: m["mesh_id"],
        )
        for _mesh in _rng.sample(_pool, min(MESHES_PER_CATEGORY, len(_pool))):
            _photos = sorted(_photos_of[_mesh["mesh_id"]], key=lambda q: q["query_id"])
            _used = _rng.sample(_photos, min(MAX_PHOTOS_PER_MESH, len(_photos)))
            sample_meshes.append(_mesh)
            sample_queries.extend(_used)
            _rows.append({"category": _category, "mesh_id": _mesh["mesh_id"], "photos_available": len(_photos), "photos_used": len(_used)})
    print("SAMPLE", {"meshes": len(sample_meshes), "queries": len(sample_queries)})
    mo.vstack([mo.md(f"**{len(sample_meshes)} meshes, {len(sample_queries)} query photos.**"), mo.ui.table(_rows, selection=None)])
    return sample_meshes, sample_queries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Renders

    The next two cells render every sampled mesh with every light setup (27 meshes x 5 setups x 25
    views = 3,375 images, about 2 minutes). Renders that exist on disk are kept, so the button is
    needed only once. When all renders exist, or when the notebook runs as a script, the renders
    are made or loaded without the button.

    | Field | Meaning | Expected |
    |:---|:---|:---|
    | `setup` | Light setup | |
    | `rendered` | Meshes rendered in this run | |
    | `already_there` | Meshes whose 25 renders existed | |
    | `seconds` | Render time of this setup in this run | |
    | `views_touching_border` | Views in which the object touched the border of the raw render (clipping) | 0 |
    """)
    return


@app.cell
def _(mo):
    render_button = mo.ui.run_button(label="Render the missing views")
    render_button
    return (render_button,)


@app.cell
def _(
    CFG,
    Image,
    LIGHTS,
    MeshRenderer,
    RENDER_DIR,
    VIEWS,
    mo,
    np,
    render_button,
    sample_meshes,
    time,
    tqdm,
):
    def render_folder(light, mesh):
        """Folder of the renders of one mesh under one light setup."""
        return RENDER_DIR / light / mesh["mesh_id"]

    def renders_exist(light, mesh):
        return all((render_folder(light, mesh) / f"view_{v:02d}.png").exists() for v in range(VIEWS))

    _missing = sum(not renders_exist(light, mesh) for light in LIGHTS for mesh in sample_meshes)
    mo.stop(
        _missing > 0 and not render_button.value and mo.app_meta().mode != "script",
        mo.md(f"{_missing} of {len(LIGHTS) * len(sample_meshes)} mesh renders are missing. Press the button to render them."),
    )

    _front = CFG["render"]["front_axis"]["pix3d"]
    _rows = []
    for _light in LIGHTS:
        _renderer = MeshRenderer.from_config(CFG["render"], lighting=_light)
        _done, _kept, _border, _start = 0, 0, 0, time.perf_counter()
        for _mesh in tqdm(sample_meshes, desc=_light, unit="mesh"):
            if renders_exist(_light, _mesh):
                _kept += 1
                continue
            _views = _renderer.render_file(_mesh["mesh_path"], front_axis=_front)
            _folder = render_folder(_light, _mesh)
            _folder.mkdir(parents=True, exist_ok=True)
            for _v, _image in enumerate(_views.images):
                Image.fromarray(_image).save(_folder / f"view_{_v:02d}.png")
            _border += int(_views.touches_border.sum())
            _done += 1
        _rows.append({"setup": _light, "rendered": _done, "already_there": _kept,
                      "seconds": round(time.perf_counter() - _start, 1), "views_touching_border": _border})

    def load_views(light, mesh):
        """The 25 renders of one mesh under one light setup: uint8 (25, 224, 224, 3)."""
        return np.stack([np.asarray(Image.open(render_folder(light, mesh) / f"view_{v:02d}.png")) for v in range(VIEWS)])

    print("RENDERS", _rows)
    mo.ui.table(_rows, selection=None)
    return (load_views,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Contact sheet

    The next cell shows four sampled meshes (from different categories), one per row. Each row
    starts with one of the mesh's query photos; then come two views for each light setup, in the
    order L0, L1, L2, L3, L4: a view from the middle ring (15° elevation, 22.5° azimuth) and one
    from the high ring (45° elevation, 90° azimuth).

    What to look for:

    - Is the 3D form clear: do the parts separate, can you see which surfaces face which way?
    - Are there areas of pure white (saturation) or pure black where the form disappears?
    - Which setup looks most like the photos, in the light and dark areas?
    """)
    return


@app.cell
def _(Image, ImageDraw, LIGHTS, load_views, paths, sample_meshes, sample_queries):
    _tile, _shown_views = 112, (8, 18)
    _rows = [m for i, m in enumerate(sample_meshes) if i % 7 == 0][:4]  # four meshes from different categories
    contact_sheet = Image.new("RGB", ((1 + len(LIGHTS) * len(_shown_views)) * _tile, 16 + len(_rows) * _tile), "white")
    _draw = ImageDraw.Draw(contact_sheet)
    _draw.text((4, 2), "photo", fill="black")
    for _i, _light in enumerate(LIGHTS):
        _draw.text(((1 + _i * len(_shown_views)) * _tile + 4, 2), _light, fill="black")
    for _r, _mesh in enumerate(_rows):
        _photo = next(q for q in sample_queries if q["mesh_id"] == _mesh["mesh_id"])
        with Image.open(paths.resolve(_photo["query_path"])) as _image:
            contact_sheet.paste(_image.resize((_tile, _tile)), (0, 16 + _r * _tile))
        for _i, _light in enumerate(LIGHTS):
            _views = load_views(_light, _mesh)
            for _j, _v in enumerate(_shown_views):
                _x = (1 + _i * len(_shown_views) + _j) * _tile
                contact_sheet.paste(Image.fromarray(_views[_v]).resize((_tile, _tile)), (_x, 16 + _r * _tile))
    print("CONTACT_SHEET", {"meshes": [m["mesh_id"] for m in _rows]})
    contact_sheet
    return (contact_sheet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Image statistics

    The next cell measures the object pixels (not the grey background) of all renders of each
    setup, and of the query photos (converted to grey) for comparison:

    | Field | Meaning |
    |:---|:---|
    | `images` | The setup (or "photos") |
    | `mean_brightness` | Average grey value of the object pixels, 0 (black) to 255 (white) |
    | `contrast` | Standard deviation of the grey values on the object: how much the shading varies. Flat images have a low value |
    | `white_percent` | Share of object pixels at 250 or more: saturated areas, where the form is lost |
    | `black_percent` | Share of object pixels at 10 or less |

    A render setup whose brightness and contrast are near those of the photos looks more like a
    photo in its light and dark areas. This is a hint, not the decision; section 7 measures the match.
    """)
    return


@app.cell
def _(BACKGROUND, Image, LIGHTS, load_views, mo, np, object_masks, paths, sample_meshes, sample_queries):
    def grey_statistics(name, images):
        """Brightness statistics of the object pixels of RGB images: uint8 (N, H, W, 3)."""
        grey = images.astype(np.float32).mean(axis=-1)
        values = grey[object_masks(images, BACKGROUND)]
        return {
            "images": name,
            "mean_brightness": round(float(values.mean()), 1),
            "contrast": round(float(values.std()), 1),
            "white_percent": round(float((values >= 250).mean() * 100), 2),
            "black_percent": round(float((values <= 10).mean() * 100), 2),
        }

    _rows = [grey_statistics(_light, np.concatenate([load_views(_light, _m) for _m in sample_meshes])) for _light in LIGHTS]
    query_images = np.stack([np.asarray(Image.open(paths.resolve(q["query_path"]))) for q in sample_queries])
    _rows.append(grey_statistics("photos", query_images))
    image_statistics = _rows
    print("STATISTICS", _rows)
    mo.ui.table(_rows, selection=None)
    return image_statistics, query_images


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Features

    The next cell turns every query photo and every render into a feature vector with the active
    backbone and feature recipe (`configs/backbone.yaml`; DINOv3 ViT-B/16, class token). This takes
    about 20 seconds. It returns:

    - query features: (queries, 768);
    - view features of each setup: (27 meshes, 25 views, 768).
    """)
    return


@app.cell
def _(BACKGROUND, CFG, FeatureExtractor, LIGHTS, VIEWS, load_views, np, object_masks, query_images, sample_meshes):
    extractor = FeatureExtractor(CFG["backbone"])
    query_features = extractor.extract(query_images, object_masks(query_images, BACKGROUND))
    view_features = {}
    for _light in LIGHTS:
        _images = np.concatenate([load_views(_light, _m) for _m in sample_meshes])  # (27 * 25, 224, 224, 3)
        _features = extractor.extract(_images, object_masks(_images, BACKGROUND))
        view_features[_light] = _features.reshape(len(sample_meshes), VIEWS, -1)
    print("FEATURES", {"backbone": extractor.spec.name, "recipe": extractor.recipe, "queries": query_features.shape,
                       "views": {k: v.shape for k, v in view_features.items()}})
    return extractor, query_features, view_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Retrieval results

    For each setup, every query photo is compared with the 27 meshes, and the meshes are ranked.
    The main score is the **view score** (the best-matching view of each mesh).

    | Field | Meaning |
    |:---|:---|
    | `setup` | Light setup |
    | `R@1` | Share of queries whose correct mesh is ranked first (%) |
    | `R@1_interval` | 95 % bootstrap interval of R@1: the range in which the true value probably lies, given this number of queries |
    | `R@5`, `R@10` | Share of queries whose correct mesh is among the first 5 or 10 (%) |
    | `Cat@1` | Share of queries whose first result has the correct category (%) |
    | `vs_L0` | R@1 of this setup minus R@1 of L0 (the v1 setup), in percentage points |
    | `vs_L0_interval` | 95 % interval of that difference (paired bootstrap: the same queries for both setups). If it does not include 0, the difference is clear |
    | `share_better_than_L0` | Share of the bootstrap resamples in which this setup beats L0 (near 1.0 = clearly better; near 0.5 = no difference) |

    The second table gives R@1 with the three scores: view score, mesh score (mean of the views),
    and the combined score with `ALPHA`. The chart shows R@1 (view score) with its interval.
    """)
    return


@app.cell
def _(ALPHA, LIGHTS, metrics, mo, np, plt, query_features, sample_meshes, sample_queries, search, view_features):
    _mesh_column = {m["mesh_id"]: i for i, m in enumerate(sample_meshes)}
    _correct = np.array([_mesh_column[q["mesh_id"]] for q in sample_queries])
    _query_categories = np.array([q["category"] for q in sample_queries])
    _mesh_categories = np.array([m["category"] for m in sample_meshes])

    hits_at_1 = {}  # per setup: True for each query whose correct mesh is first (view score)
    results, score_rows = [], []
    for _light in LIGHTS:
        _views = view_features[_light]
        _view_score = search.view_scores(query_features, _views)
        _ranks = metrics.correct_ranks(_view_score, _correct)
        hits_at_1[_light] = _ranks <= 1
        _low, _high = metrics.bootstrap_interval(hits_at_1[_light])
        results.append({
            "setup": _light,
            "R@1": round(100 * metrics.recall_at_k(_ranks, 1), 1),
            "R@1_interval": f"{100 * _low:.1f} to {100 * _high:.1f}",
            "R@5": round(100 * metrics.recall_at_k(_ranks, 5), 1),
            "R@10": round(100 * metrics.recall_at_k(_ranks, 10), 1),
            "Cat@1": round(100 * metrics.category_hits(_view_score, _query_categories, _mesh_categories).mean(), 1),
        })
        score_rows.append({
            "setup": _light,
            "R@1_view_score": round(100 * metrics.recall_at_k(_ranks, 1), 1),
            "R@1_mesh_score": round(100 * metrics.recall_at_k(metrics.correct_ranks(search.mesh_scores(query_features, _views), _correct), 1), 1),
            f"R@1_combined_alpha_{ALPHA}": round(100 * metrics.recall_at_k(metrics.correct_ranks(search.combined_scores(query_features, _views, ALPHA), _correct), 1), 1),
        })
    for _row in results:
        _comparison = metrics.paired_difference(hits_at_1["L0"].astype(float), hits_at_1[_row["setup"]].astype(float))
        _row["vs_L0"] = round(100 * _comparison["difference"], 1)
        _row["vs_L0_interval"] = f"{100 * _comparison['low']:.1f} to {100 * _comparison['high']:.1f}"
        _row["share_better_than_L0"] = round(_comparison["share_b_better"], 3)

    chart, _axis = plt.subplots(figsize=(6, 3))
    _values = [r["R@1"] for r in results]
    _intervals = [tuple(float(x) for x in r["R@1_interval"].split(" to ")) for r in results]
    _axis.bar(LIGHTS, _values, color="#888888")
    _axis.errorbar(LIGHTS, _values, yerr=[[v - lo for v, (lo, _) in zip(_values, _intervals)], [hi - v for v, (_, hi) in zip(_values, _intervals)]],
                   fmt="none", ecolor="black", capsize=4)
    _axis.set_ylabel("R@1 (%), view score")
    _axis.set_title(f"Zero-shot R@1 on {len(sample_meshes)} meshes, {len(sample_queries)} photos")
    chart.tight_layout()
    print("RESULTS", results)
    print("SCORES", score_rows)
    mo.vstack([mo.ui.table(results, selection=None), mo.ui.table(score_rows, selection=None), chart])
    return chart, hits_at_1, results, score_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. R@1 per category

    The next cell gives R@1 (view score, %) for each category and setup. A setup can help some
    kinds of objects (for example flat-panelled wardrobes) more than others (thin-legged chairs).
    The categories have few queries each, so single numbers here are noisy.
    """)
    return


@app.cell
def _(LIGHTS, hits_at_1, mo, np, sample_queries):
    _categories = np.array([q["category"] for q in sample_queries])
    category_rows = []
    for _category in sorted(set(_categories)):
        _selected = _categories == _category
        category_rows.append({"category": _category, "queries": int(_selected.sum()),
                              **{_light: round(100 * float(hits_at_1[_light][_selected].mean()), 1) for _light in LIGHTS}})
    print("PER_CATEGORY", category_rows)
    mo.ui.table(category_rows, selection=None)
    return (category_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Save the results

    The button writes the results into a new run folder, `runs/<date>_pix3d_<backbone>_lighting_test/`:

    - `config.yaml`: all settings of this test;
    - `metrics.json`: the sample, the image statistics, and all result tables;
    - `contact_sheet.png` and `r1_chart.png`.

    When the notebook runs as a script, the results are saved without the button. The renders
    in `data/derived/renders_lighting_test/` stay until the light setup is chosen.
    """)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save the results")
    save_button
    return (save_button,)


@app.cell
def _(
    ALPHA,
    CFG,
    MAX_PHOTOS_PER_MESH,
    MESHES_PER_CATEGORY,
    SEED,
    category_rows,
    chart,
    contact_sheet,
    datetime,
    extractor,
    git_state,
    image_statistics,
    json,
    make_run_name,
    mo,
    paths,
    results,
    sample_meshes,
    sample_queries,
    save_button,
    save_config,
    score_rows,
):
    mo.stop(not save_button.value and mo.app_meta().mode != "script", mo.md("Press the button to save the results."))
    _name = make_run_name("pix3d", extractor.spec.name, "lighting_test")
    _run_dir = paths.ensure_dir(paths.RUNS_DIR / _name)
    save_config(CFG, _run_dir / "config.yaml")
    _commit, _dirty = git_state()
    _record = {
        "run_name": _name,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _commit,
        "git_dirty": _dirty,
        "backbone": extractor.spec.name,
        "feature_recipe": extractor.recipe,
        "sample": {"seed": SEED, "meshes_per_category": MESHES_PER_CATEGORY, "max_photos_per_mesh": MAX_PHOTOS_PER_MESH,
                   "meshes": [m["mesh_id"] for m in sample_meshes], "queries": [q["query_id"] for q in sample_queries]},
        "alpha": ALPHA,
        "image_statistics": image_statistics,
        "results_view_score": results,
        "r1_by_score": score_rows,
        "r1_by_category": category_rows,
    }
    (_run_dir / "metrics.json").write_text(json.dumps(_record, indent=2) + "\n")
    contact_sheet.save(_run_dir / "contact_sheet.png")
    chart.savefig(_run_dir / "r1_chart.png", dpi=120)
    print("SAVED", str(_run_dir.relative_to(paths.ROOT)))
    mo.md(f"Saved to `{_run_dir.relative_to(paths.ROOT)}`.")
    return


if __name__ == "__main__":
    app.run()
