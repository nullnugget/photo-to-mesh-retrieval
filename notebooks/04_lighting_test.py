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
    # 04: Lighting test

    **Question:** which light setup makes the grey renders look most like the photos, to the
    backbone? The renders are the only view of a mesh that the model gets, and in a grey render
    the shading is the only cue to the 3D form. The light setup decides that shading.

    **Method** (no training needed):

    1. Render every Pix3D mesh (395) from all 25 cameras with each light setup L0 to L4
       (`configs/render.yaml`), and turn each render straight into a DINOv3 feature vector. Only the
       features are kept (about 15 MB per setup), not the images.
    2. Turn every query photo into a feature vector in the same way.
    3. For each light setup, search all 395 meshes with every query photo and measure how often the
       correct mesh comes first (R@1). The frozen backbone already places photos and renders in one
       feature space, so this "zero-shot" search works before any training.
    4. Compare the light setups with confidence intervals, and look at the renders of a small
       sample of meshes (3 per category, saved as images) by eye.

    | Setup | Light |
    |:---|:---|
    | L0 | One point light at the camera (the v1 setup) |
    | L1 | The same position, with less ambient light and smaller highlights |
    | L2 | A key light 35° above and 35° left of the viewing direction (moves with the camera) |
    | L3 | L2 plus a weaker fill light from the right |
    | L4 | One fixed light from straight above, like a ceiling light |
    | L2_bright | L2, with the brightness moved up (146 to 255): the object is always lighter than the background |
    | L2_dark | L2, with the brightness moved down (14 to 112): the object is always darker than the background |

    The two L2 variants test the separation from the background: with L2, surfaces at about 50° to
    the light have exactly the background grey (128), so parts of the outline blend into the
    background, much more often than in the photos.

    **Queries:** all train and val photos of Pix3D (8,557; no training happens here, so train
    photos are not "seen"). The test photos stay out, so that the choice of the light does not use
    the test set.

    **Why the full gallery:** a first run with a gallery of only 27 meshes could not separate the
    setups (R@5 was about 99.5 % for all of them; each photo competed with only 2 meshes of its own
    category). With all 395 meshes, each photo competes with every look-alike mesh, as in real use.

    **Limit:** the ranking is for the frozen backbone; after training the heads, it could change,
    so a short training run checks the best one or two setups later.

    **Outputs:**

    - sample renders: `data/derived/renders_lighting_test/<setup>/<mesh id>/view_00.png` ... `view_24.png`;
    - gallery features: `data/derived/renders_lighting_test/gallery_features_<setup>.npz`
      (both temporary; deleted after the decision);
    - results: `runs/<date>_pix3d_<backbone>_lighting_test/` (settings, metrics, contact sheets),
      written by the last section.

    **How to run:** `marimo edit notebooks/04_lighting_test.py`, then press the buttons in sections 3,
    7, and 10. As a script (`python notebooks/04_lighting_test.py`), all steps run and the results are
    saved. The gallery features take about 25 minutes the first time; later runs load them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules, and sets the experiment settings:

    | Setting | Value | Meaning |
    |:---|:---|:---|
    | gallery (fixed) | all Pix3D meshes (395) | The meshes that each query is searched against, for the retrieval results |
    | queries (fixed) | all non-test Pix3D photos (8,557) | The photos that are searched with, for the retrieval results |
    | `SAMPLE_MESHES_PER_CATEGORY` | 3 | Size of the small sample (27 meshes) that is saved as images, only for the contact sheet and the image statistics |
    | `SAMPLE_MIN_PHOTOS` | 3 | A mesh of the small sample needs at least this many non-test photos |
    | `SAMPLE_PHOTOS_PER_MESH` | 30 | Photos per sample mesh used for the image statistics (at most) |
    | `SEED` | 0 | Fixed seed of the random sample |
    | `QUERY_CHUNK` | 512 | Query photos read and turned into features at once (limits the memory) |
    | `LIGHTS` | L0 ... L4, L2_bright, L2_dark | The light setups to compare (all presets in `configs/render.yaml`) |
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
    from scipy import ndimage
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
    SAMPLE_MESHES_PER_CATEGORY = 3
    SAMPLE_MIN_PHOTOS = 3
    SAMPLE_PHOTOS_PER_MESH = 30
    SEED = 0
    QUERY_CHUNK = 512
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
        SAMPLE_PHOTOS_PER_MESH,
        SAMPLE_MESHES_PER_CATEGORY,
        SAMPLE_MIN_PHOTOS,
        MeshRenderer,
        QUERY_CHUNK,
        RENDER_DIR,
        SEED,
        VIEWS,
        datetime,
        git_state,
        json,
        load_split,
        make_run_name,
        metrics,
        ndimage,
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
    ## 2. Gallery, queries, and the sample for the contact sheet

    The next cell:

    1. reads the Pix3D meshes (the gallery, 395) and queries from the catalog, and the test part of
       the `seen_meshes` split (these photos are never used here); the queries are all other photos;
    2. samples 3 random meshes per category (among those with at least 3 non-test photos) and up to
       30 of their photos, for the contact sheet and the image statistics.

    The table lists the sample:

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
    SAMPLE_PHOTOS_PER_MESH,
    SAMPLE_MESHES_PER_CATEGORY,
    SAMPLE_MIN_PHOTOS,
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
    gallery_meshes = sorted(_meshes, key=lambda m: m["mesh_id"])
    all_queries = sorted(_queries, key=lambda q: q["query_id"])

    _rng = random.Random(SEED)
    sample_meshes, sample_queries, _rows = [], [], []
    for _category in sorted({m["category"] for m in _meshes}):
        _pool = sorted(
            (m for m in _meshes if m["category"] == _category and len(_photos_of.get(m["mesh_id"], [])) >= SAMPLE_MIN_PHOTOS),
            key=lambda m: m["mesh_id"],
        )
        for _mesh in _rng.sample(_pool, min(SAMPLE_MESHES_PER_CATEGORY, len(_pool))):
            _photos = sorted(_photos_of[_mesh["mesh_id"]], key=lambda q: q["query_id"])
            _used = _rng.sample(_photos, min(SAMPLE_PHOTOS_PER_MESH, len(_photos)))
            sample_meshes.append(_mesh)
            sample_queries.extend(_used)
            _rows.append({"category": _category, "mesh_id": _mesh["mesh_id"], "photos_available": len(_photos), "photos_used": len(_used)})
    print("SAMPLE", {"gallery": len(gallery_meshes), "queries": len(all_queries), "sample_meshes": len(sample_meshes), "sample_photos": len(sample_queries)})
    mo.vstack([
        mo.md(f"**Gallery: {len(gallery_meshes)} meshes. Queries: {len(all_queries)} photos.** "
              f"Sample for the contact sheet: {len(sample_meshes)} meshes, {len(sample_queries)} photos."),
        mo.ui.table(_rows, selection=None),
    ])
    return all_queries, gallery_meshes, sample_meshes, sample_queries


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sample renders

    The next two cells render the sampled meshes with every light setup and save the images
    (27 meshes x 5 setups x 25 views = 3,375 images, about 2 minutes), for the contact sheet and
    the image statistics. Renders that exist on disk are kept, so the button is
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
    order of the table in the introduction: a view from the middle ring (15° elevation, 22.5° azimuth) and one
    from the high ring (45° elevation, 90° azimuth).

    What to look for:

    - Is the 3D form clear: do the parts separate, can you see which surfaces face which way?
    - Are there areas of pure white (saturation) or pure black where the form disappears?
    - Which setup looks most like the photos, in the light and dark areas?
    - Does the outline stand out from the grey background everywhere?
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

    The next cell measures the object pixels (not the grey background) of the sample renders of
    each setup, and of the sample photos (converted to grey) for comparison:

    | Field | Meaning |
    |:---|:---|
    | `images` | The setup (or "photos") |
    | `mean_brightness` | Average grey value of the object pixels, 0 (black) to 255 (white) |
    | `contrast` | Standard deviation of the grey values on the object: how much the shading varies. Flat images have a low value |
    | `white_percent` | Share of object pixels at 250 or more: saturated areas, where the form is lost |
    | `black_percent` | Share of object pixels at 10 or less |
    | `near_background_percent` | Share of object pixels within 8 grey levels of the background (128): there, the object and the background look the same |
    | `outline_difference` | Mean difference from the background grey along the outline (a band of 2 pixels inside the object edge). Higher = the outline stands out more |

    (The object pixels are found as the pixels that differ from the background grey, so an object
    pixel of exactly 128 is not counted; this makes `near_background_percent` slightly too low for
    the setups whose range includes 128.)

    A render setup whose brightness and contrast are near those of the photos looks more like a
    photo in its light and dark areas. This is a hint, not the decision; section 7 measures the match.
    """)
    return


@app.cell
def _(BACKGROUND, Image, LIGHTS, load_views, mo, ndimage, np, object_masks, paths, sample_meshes, sample_queries):
    def grey_statistics(name, images):
        """Brightness statistics of the object pixels of RGB images: uint8 (N, H, W, 3)."""
        grey = images.astype(np.float32).mean(axis=-1)
        masks = object_masks(images, BACKGROUND)
        values = grey[masks]
        # The outline band: object pixels that a 2-pixel erosion removes (per image, not across images).
        outline = masks & ~ndimage.binary_erosion(masks, structure=np.ones((1, 3, 3), bool), iterations=2)
        return {
            "images": name,
            "mean_brightness": round(float(values.mean()), 1),
            "contrast": round(float(values.std()), 1),
            "white_percent": round(float((values >= 250).mean() * 100), 2),
            "black_percent": round(float((values <= 10).mean() * 100), 2),
            "near_background_percent": round(float((np.abs(values - BACKGROUND[0]) <= 8).mean() * 100), 1),
            "outline_difference": round(float(np.abs(grey[outline] - BACKGROUND[0]).mean()), 1),
        }

    _rows = [grey_statistics(_light, np.concatenate([load_views(_light, _m) for _m in sample_meshes])) for _light in LIGHTS]
    _rows.append(grey_statistics("photos", np.stack([np.asarray(Image.open(paths.resolve(q["query_path"]))) for q in sample_queries])))
    image_statistics = _rows
    print("STATISTICS", _rows)
    mo.ui.table(_rows, selection=None)
    return (image_statistics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Query features

    The next cell turns every query photo into a feature vector with the active backbone and
    feature recipe (`configs/backbone.yaml`; DINOv3 ViT-B/16, class token). It reads the photos in
    chunks of 512, so that memory stays small, and takes about 40 seconds. Result: (8,557, 768).
    """)
    return


@app.cell
def _(BACKGROUND, CFG, FeatureExtractor, Image, QUERY_CHUNK, all_queries, np, object_masks, paths, tqdm):
    extractor = FeatureExtractor(CFG["backbone"])
    _parts = []
    for _start in tqdm(range(0, len(all_queries), QUERY_CHUNK), desc="query features", unit="chunk"):
        _images = np.stack([np.asarray(Image.open(paths.resolve(q["query_path"]))) for q in all_queries[_start : _start + QUERY_CHUNK]])
        _parts.append(extractor.extract(_images, object_masks(_images, BACKGROUND)))
    query_features = np.concatenate(_parts)
    print("QUERY_FEATURES", {"backbone": extractor.spec.name, "recipe": extractor.recipe, "shape": query_features.shape})
    return extractor, query_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Gallery features

    The next two cells render every gallery mesh with every light setup and turn the 25 views
    straight into feature vectors; the images are not saved. The features of each setup are saved
    in `gallery_features_<setup>.npz` (395 x 25 x 768 numbers, float16, about 15 MB) and loaded in
    later runs, if they were made with the same meshes, backbone, and recipe. The first run takes
    about 25 minutes (5 setups x 395 meshes); the button starts it.

    | Field | Meaning | Expected |
    |:---|:---|:---|
    | `setup` | Light setup | |
    | `source` | "computed" in this run, or "loaded" from the file | |
    | `minutes` | Time to compute the features of this setup | |
    | `views_touching_border` | Views in which the object touched the border of the raw render | 0 |
    """)
    return


@app.cell
def _(mo):
    gallery_button = mo.ui.run_button(label="Compute the missing gallery features")
    gallery_button
    return (gallery_button,)


@app.cell
def _(
    BACKGROUND,
    CFG,
    LIGHTS,
    MeshRenderer,
    RENDER_DIR,
    VIEWS,
    extractor,
    gallery_button,
    gallery_meshes,
    mo,
    np,
    object_masks,
    time,
    tqdm,
):
    _mesh_ids = np.array([m["mesh_id"] for m in gallery_meshes])

    def gallery_file(light):
        return RENDER_DIR / f"gallery_features_{light}.npz"

    def saved_features(light):
        """The saved features of one setup, or None if missing or made with other meshes, backbone, or recipe."""
        if not gallery_file(light).exists():
            return None
        saved = np.load(gallery_file(light))
        same = (np.array_equal(saved["mesh_ids"], _mesh_ids) and str(saved["backbone"]) == extractor.spec.name
                and str(saved["recipe"]) == extractor.recipe)
        return saved["features"] if same else None

    _missing = [light for light in LIGHTS if saved_features(light) is None]
    mo.stop(
        bool(_missing) and not gallery_button.value and mo.app_meta().mode != "script",
        mo.md(f"Gallery features missing for {', '.join(_missing)} (about 5 minutes per setup). Press the button."),
    )

    _front = CFG["render"]["front_axis"]["pix3d"]
    view_features, _rows = {}, []
    for _light in LIGHTS:
        _features = saved_features(_light)
        if _features is not None:
            view_features[_light] = _features
            _rows.append({"setup": _light, "source": "loaded", "minutes": 0.0, "views_touching_border": "n/a"})
            continue
        _renderer = MeshRenderer.from_config(CFG["render"], lighting=_light)
        _features = np.zeros((len(gallery_meshes), VIEWS, extractor.dim), dtype=np.float16)
        _border, _start = 0, time.perf_counter()
        for _i, _mesh in enumerate(tqdm(gallery_meshes, desc=f"gallery {_light}", unit="mesh")):
            _views = _renderer.render_file(_mesh["mesh_path"], front_axis=_front)
            _features[_i] = extractor.extract(_views.images, object_masks(_views.images, BACKGROUND))
            _border += int(_views.touches_border.sum())
        np.savez(gallery_file(_light), features=_features, mesh_ids=_mesh_ids, backbone=extractor.spec.name, recipe=extractor.recipe)
        view_features[_light] = _features
        _rows.append({"setup": _light, "source": "computed", "minutes": round((time.perf_counter() - _start) / 60, 1), "views_touching_border": _border})
    print("GALLERY", _rows)
    mo.ui.table(_rows, selection=None)
    return (view_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Retrieval results

    For each setup, every query photo is compared with all 395 meshes, and the meshes are ranked.
    The main score is the **view score** (the best-matching view of each mesh).

    | Field | Meaning |
    |:---|:---|
    | `setup` | Light setup |
    | `R@1` | Share of queries whose correct mesh is ranked first (%) |
    | `R@1_interval` | 95 % bootstrap interval of R@1: the range in which the true value probably lies, given this number of queries |
    | `R@5`, `R@10` | Share of queries whose correct mesh is among the first 5 or 10 (%) |
    | `Cat@1` | Share of queries whose first result has the correct category (%) |
    | `vs_baseline` | R@1 of this setup minus R@1 of the baseline, in percentage points. The baseline is the active setup in `configs/render.yaml` (`render.lighting`) |
    | `vs_baseline_interval` | 95 % interval of that difference (paired bootstrap: the same queries for both setups). If it does not include 0, the difference is clear |
    | `share_better_than_baseline` | Share of the bootstrap resamples in which this setup beats the baseline (near 1.0 = clearly better; near 0.5 = no difference; near 0.0 = clearly worse) |

    The second table gives R@1 with the three scores: view score, mesh score (mean of the views),
    and the combined score with `ALPHA`. The chart shows R@1 (view score) with its interval.
    """)
    return


@app.cell
def _(ALPHA, CFG, LIGHTS, all_queries, gallery_meshes, metrics, mo, np, plt, query_features, search, view_features):
    _mesh_column = {m["mesh_id"]: i for i, m in enumerate(gallery_meshes)}
    _correct = np.array([_mesh_column[q["mesh_id"]] for q in all_queries])
    _query_categories = np.array([q["category"] for q in all_queries])
    _mesh_categories = np.array([m["category"] for m in gallery_meshes])

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
    _baseline = CFG["render"]["lighting"]
    for _row in results:
        _comparison = metrics.paired_difference(hits_at_1[_baseline].astype(float), hits_at_1[_row["setup"]].astype(float))
        _row["vs_baseline"] = round(100 * _comparison["difference"], 1)
        _row["vs_baseline_interval"] = f"{100 * _comparison['low']:.1f} to {100 * _comparison['high']:.1f}"
        _row["share_better_than_baseline"] = round(_comparison["share_b_better"], 3)

    chart, _axis = plt.subplots(figsize=(7, 3.2))
    _values = [r["R@1"] for r in results]
    _intervals = [tuple(float(x) for x in r["R@1_interval"].split(" to ")) for r in results]
    _axis.bar(LIGHTS, _values, color="#888888")
    _axis.tick_params(axis="x", labelrotation=30)
    _axis.errorbar(LIGHTS, _values, yerr=[[v - lo for v, (lo, _) in zip(_values, _intervals)], [hi - v for v, (_, hi) in zip(_values, _intervals)]],
                   fmt="none", ecolor="black", capsize=4)
    _axis.set_ylabel("R@1 (%), view score")
    _axis.set_title(f"Zero-shot R@1: {len(all_queries)} photos against {len(gallery_meshes)} meshes")
    chart.tight_layout()
    print("RESULTS", results)
    print("SCORES", score_rows)
    mo.vstack([mo.ui.table(results, selection=None), mo.ui.table(score_rows, selection=None), chart])
    return chart, hits_at_1, results, score_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. R@1 per category

    The next cell gives R@1 (view score, %) for each category and setup. A setup can help some
    kinds of objects (for example flat-panelled wardrobes) more than others (thin-legged chairs).
    Categories with few queries (misc, tool) give noisy numbers.
    """)
    return


@app.cell
def _(LIGHTS, all_queries, hits_at_1, mo, np):
    _categories = np.array([q["category"] for q in all_queries])
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
    ## 10. Save the results

    The button writes the results into a new run folder, `runs/<date>_pix3d_<backbone>_lighting_test/`:

    - `config.yaml`: all settings of this test;
    - `metrics.json`: the sample, the image statistics, and all result tables;
    - `contact_sheet.png` and `r1_chart.png`.

    When the notebook runs as a script, the results are saved without the button. The sample
    renders and the gallery features in `data/derived/renders_lighting_test/` stay until the light
    setup is chosen.
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
    SAMPLE_PHOTOS_PER_MESH,
    SAMPLE_MESHES_PER_CATEGORY,
    SEED,
    category_rows,
    chart,
    contact_sheet,
    datetime,
    all_queries,
    extractor,
    gallery_meshes,
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
        "gallery_meshes": len(gallery_meshes),
        "queries": len(all_queries),
        "query_set": "pix3d seen_meshes train + val (test excluded)",
        "sample": {"seed": SEED, "meshes_per_category": SAMPLE_MESHES_PER_CATEGORY, "max_photos_per_mesh": SAMPLE_PHOTOS_PER_MESH,
                   "meshes": [m["mesh_id"] for m in sample_meshes], "photos": [q["query_id"] for q in sample_queries]},
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
