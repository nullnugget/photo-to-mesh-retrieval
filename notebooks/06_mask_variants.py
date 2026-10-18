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
    # 06: Mask Variants

    With SAM 2.1 small masks, zero-shot R@1 on the Pix3D tuning photos drops from 67.7 % (Pix3D masks)
    to 57.1 %. This experiment tests two ways to close the gap, on the same photos and the same gallery:

    **A. Better masks.** SAM small or SAM large, each with SAM's one-mask output ("single") or the largest
    of its three candidate masks ("largest"). The larger model is more likely to segment the whole object
    instead of a part (the frame of a chair without its cushion).

    **B. Queries that tolerate mask errors.** The normal query keeps only the pixels inside the mask and
    paints everything else grey. If the mask misses a part, that part is gone from the query. Two
    variants cut out less strictly:

    - "grown": the SAM mask grown by 2 % of the box size in every direction (kept inside the box), so a
      missed thin part (a leg, the edge of a cushion) is often covered again, at the price of a thin ring
      of real background;
    - "box only": no mask at all; the query keeps every pixel inside the box, background included. This
      shows what the mask is worth: if "box only" is close to the SAM masks, the mask helps little.

    | Variant | Mask | Query pixels |
    |:---|:---|:---|
    | `reference` | Pix3D mask | inside the mask (the normal query; the upper bound) |
    | `small_single` | SAM small, one-mask output | inside the mask (the current setting) |
    | `small_largest` | SAM small, largest candidate | inside the mask |
    | `large_single` | SAM large, one-mask output | inside the mask |
    | `large_largest` | SAM large, largest candidate | inside the mask |
    | `small_single_grown` | SAM small, one-mask output, grown | inside the grown mask |
    | `large_single_grown` | SAM large, one-mask output, grown | inside the grown mask |
    | `box_only` | none | inside the box |

    Every SAM variant gets the box around the Pix3D mask (as a user would draw it in the demo). All query
    images are made with the same crop function as the query step, so the only difference is the mask.

    **Photos:** 1,500 random photos of the tuning set (train and val of `seen_meshes`); the test photos are
    not used. **Gallery:** the 395 Pix3D meshes (renders with L2_dark). **Search:** frozen DINOv3, view score.

    **How to run:** `marimo edit notebooks/06_mask_variants.py`, then the buttons; or
    `python notebooks/06_mask_variants.py` (as a script; about 40 minutes on the GPU; the results are saved
    in `runs/`). Needs the SAM 2.1 small and large weights (`scripts/00_download_weights.py --models
    segmenter segmenter_large`), the DINOv3 weights, and the Pix3D renders.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules, and sets the experiment settings:

    | Setting | Value | Meaning |
    |:---|:---|:---|
    | `SUBSET_SIZE` | 1,500 | Tuning photos in the experiment (random, fixed seed) |
    | `SEED` | 0 | Seed of the random choice of photos |
    | `GROW_FRACTION` | 0.02 | Growth of the "grown" masks: this share of the longer box side, in every direction |
    | `FEATURE_BATCH` | 256 | Query images per feature batch |
    | `EXAMPLE_PHOTOS` | 8 | Photos shown in the example sheet (section 6) |
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
    import torch
    from PIL import Image, ImageDraw
    from tqdm import tqdm

    # The notebook is in <repository>/notebooks/, so the repository root is one level up.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes the project package "src" importable
    from src import paths  # noqa: E402
    from src.config import load_config, save_config  # noqa: E402
    from src.data.catalog import read_table  # noqa: E402
    from src.data.splits import load_split  # noqa: E402
    from src.models.backbones import FeatureExtractor, object_masks  # noqa: E402
    from src.preprocess import segment  # noqa: E402
    from src.preprocess.crop import crop_to_square  # noqa: E402
    from src.retrieval import metrics, search  # noqa: E402
    from src.runs import git_state, make_run_name  # noqa: E402

    CFG = load_config()
    BACKGROUND = tuple(CFG["data"]["background_rgb"])
    SUBSET_SIZE = 1500
    SEED = 0
    GROW_FRACTION = 0.02
    FEATURE_BATCH = 256
    EXAMPLE_PHOTOS = 8
    VIEWS = 25
    VARIANTS = ["reference", "small_single", "small_largest", "large_single", "large_largest",
                "small_single_grown", "large_single_grown", "box_only"]
    return (
        BACKGROUND,
        CFG,
        Counter,
        EXAMPLE_PHOTOS,
        FEATURE_BATCH,
        FeatureExtractor,
        GROW_FRACTION,
        Image,
        ImageDraw,
        SEED,
        SUBSET_SIZE,
        VARIANTS,
        VIEWS,
        crop_to_square,
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
        segment,
        time,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The photos

    The next cell takes `SUBSET_SIZE` random photos from the tuning set and shows how many come from each
    category. A random choice keeps roughly the category shares of the tuning set (chairs are about 38 %).
    """)
    return


@app.cell
def _(Counter, SEED, SUBSET_SIZE, load_split, mo, paths, random, read_table):
    _parts = load_split(paths.split_file("pix3d", "seen_meshes"))["parts"]
    _tuning = set(_parts["train"]) | set(_parts["val"])
    _rows = [q for q in read_table(paths.QUERIES_CSV) if q["query_id"] in _tuning]
    PHOTOS = sorted(random.Random(SEED).sample(_rows, SUBSET_SIZE), key=lambda q: q["query_id"])
    _counts = [{"category": c, "photos": n} for c, n in sorted(Counter(q["category"] for q in PHOTOS).items())]
    print("PHOTOS", {"tuning_photos": len(_rows), "subset": len(PHOTOS)})
    mo.ui.table(_counts, selection=None)
    return (PHOTOS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Masks, query images, and query features

    For every photo, the next cells:

    1. read the photo and its Pix3D mask, and take the box around the mask;
    2. let SAM small and SAM large each encode the photo once and give both masks ("single", "largest");
    3. grow the two "single" masks, and make the box-only "mask";
    4. compare every SAM mask with the Pix3D mask (IoU and edge-tolerant IoU, as in the segment step);
    5. make the 224 × 224 query image of every variant (`crop_to_square`, as the query step) and turn the
       query images into feature vectors in batches. The query images are not saved.

    This takes about 40 minutes (SAM large needs about 0.5 s per photo). The button starts it; as a
    script it runs without the button.

    | Field | Meaning |
    |:---|:---|
    | `model` | SAM small or large |
    | `seconds_per_photo` | Time of SAM for one photo (encoding once, both masks) |
    | `peak_gpu_mib` | Largest GPU memory used by PyTorch during the run (all models together) |
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Make the masks and the query features (about 40 minutes)")
    run_button
    return (run_button,)


@app.cell
def _(
    BACKGROUND,
    CFG,
    EXAMPLE_PHOTOS,
    FEATURE_BATCH,
    FeatureExtractor,
    GROW_FRACTION,
    Image,
    PHOTOS,
    VARIANTS,
    crop_to_square,
    mo,
    np,
    object_masks,
    paths,
    run_button,
    segment,
    time,
    torch,
    tqdm,
):
    mo.stop(not run_button.value and mo.app_meta().mode != "script", mo.md("Press the button to run section 3."))
    _seg_cfg = CFG["segment"]
    _models = {"small": segment.Segmenter(_seg_cfg["segmenter"]), "large": segment.Segmenter(_seg_cfg["segmenter_large"])}
    extractor = FeatureExtractor(CFG["backbone"])
    torch.cuda.reset_peak_memory_stats()

    _seconds = {name: 0.0 for name in _models}
    _buffers = {v: [] for v in VARIANTS}  # query images waiting for the next feature batch
    _features = {v: [] for v in VARIANTS}
    IOU = {v: [] for v in VARIANTS if v not in ("reference", "box_only")}
    EXAMPLES = []  # query images of the first EXAMPLE_PHOTOS photos, for section 6

    def _flush(force=False):
        """Extract the features of the buffered query images (every FEATURE_BATCH photos, and at the end)."""
        if not force and len(_buffers["reference"]) < FEATURE_BATCH:
            return
        for variant, images in _buffers.items():
            if images:
                batch = np.stack(images)
                _features[variant].append(extractor.extract(batch, object_masks(batch, BACKGROUND)))
                images.clear()

    for _index, _photo in enumerate(tqdm(PHOTOS, desc="mask variants", unit="photo")):
        with Image.open(paths.resolve(_photo["photo_path"])) as _file:
            _image = np.asarray(_file.convert("RGB")).copy()
        _reference = segment.read_mask(_photo["mask_path"])
        _box = segment.box_of_mask(_reference)
        _masks = {"reference": _reference}
        for _name, _model in _models.items():
            _start = time.perf_counter()
            for _choice, _result in _model.masks_for_each_choice(_image, _box).items():
                _masks[f"{_name}_{_choice}"] = _result.mask
            _seconds[_name] += time.perf_counter() - _start
        _pixels = round(GROW_FRACTION * max(_box[2] - _box[0], _box[3] - _box[1]))
        for _name in _models:
            _masks[f"{_name}_single_grown"] = segment.grow_mask(_masks[f"{_name}_single"], _pixels, _box)
        _masks["box_only"] = segment.box_mask(_reference.shape, _box)

        _band = segment.edge_band(_reference, segment.edge_band_width(_reference, _seg_cfg["check"]["edge_band_fraction"]))
        for _variant in IOU:
            IOU[_variant].append((segment.mask_iou(_masks[_variant], _reference), segment.mask_iou(_masks[_variant], _reference, _band)))
        _queries = {}
        for _variant in VARIANTS:
            _mask = _masks[_variant]
            if not _mask.any():  # an empty SAM mask: fall back to the box, so that every photo has a query
                _mask = _masks["box_only"]
            _queries[_variant] = crop_to_square(_image, _mask, BACKGROUND, CFG["data"]["crop_margin"], CFG["data"]["image_size"])
            _buffers[_variant].append(_queries[_variant])
        if _index < EXAMPLE_PHOTOS:
            EXAMPLES.append((_photo["query_id"], _queries))
        _flush()
    _flush(force=True)

    QUERY_FEATURES = {v: np.concatenate(parts) for v, parts in _features.items()}
    TIMING = [{"model": name, "seconds_per_photo": round(s / len(PHOTOS), 3),
               "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20)} for name, s in _seconds.items()]
    print("TIMING", TIMING)
    mo.ui.table(TIMING, selection=None)
    return EXAMPLES, IOU, QUERY_FEATURES, TIMING, extractor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Gallery features

    The next cell turns the 25 saved views of each of the 395 Pix3D meshes into feature vectors with the
    same backbone (about 40 seconds). Result: (395, 25, 768).
    """)
    return


@app.cell
def _(BACKGROUND, FEATURE_BATCH, Image, VIEWS, extractor, np, object_masks, paths, read_table, tqdm):
    GALLERY = [m for m in read_table(paths.MESHES_CSV) if m["dataset"] == "pix3d"]
    _files = [paths.resolve(m["render_dir"]) / f"view_{v:02d}.png" for m in GALLERY for v in range(VIEWS)]
    _parts = []
    for _start in tqdm(range(0, len(_files), FEATURE_BATCH), desc="gallery features", unit="batch"):
        _images = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in _files[_start : _start + FEATURE_BATCH]])
        _parts.append(extractor.extract(_images, object_masks(_images, BACKGROUND)))
    VIEW_FEATURES = np.concatenate(_parts).reshape(len(GALLERY), VIEWS, -1)
    print("GALLERY", VIEW_FEATURES.shape)
    return GALLERY, VIEW_FEATURES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Results

    | Field | Meaning |
    |:---|:---|
    | `variant` | See the table at the top |
    | `R@1`, `R@5` | Share of photos whose correct mesh is ranked first, or among the first 5 (%) |
    | `R@1_interval` | 95 % bootstrap interval of R@1 |
    | `vs_reference` | R@1 minus R@1 with the Pix3D masks, in points (the target was at least −3.0) |
    | `vs_small_single` | R@1 minus R@1 with the current setting (SAM small, single), in points |
    | `vs_small_single_interval` | 95 % interval of that difference (paired: the same photos). If it does not include 0, the difference is clear |
    | `median_iou`, `median_iou_edge_tolerant` | Agreement of the mask with the Pix3D mask (not for `reference` and `box_only`) |
    | `share_iou_below_0.70` | Share of photos whose mask has an IoU below 0.70 |

    The second table gives R@1 by Pix3D flag: `occluded` photos (another object hides part of the object)
    lost the most with SAM small. The chart shows R@1 with its interval.
    """)
    return


@app.cell
def _(GALLERY, IOU, PHOTOS, QUERY_FEATURES, VARIANTS, VIEW_FEATURES, metrics, mo, np, plt, search):
    _column = {m["mesh_id"]: i for i, m in enumerate(GALLERY)}
    _correct = np.array([_column[q["mesh_id"]] for q in PHOTOS])
    HITS = {}
    RESULTS = []
    for _variant in VARIANTS:
        _ranks = metrics.correct_ranks(search.view_scores(QUERY_FEATURES[_variant], VIEW_FEATURES), _correct)
        HITS[_variant] = _ranks <= 1
        _low, _high = metrics.bootstrap_interval(HITS[_variant])
        _row = {"variant": _variant, "R@1": round(100 * metrics.recall_at_k(_ranks, 1), 1),
                "R@1_interval": f"{100 * _low:.1f} to {100 * _high:.1f}", "R@5": round(100 * metrics.recall_at_k(_ranks, 5), 1)}
        if _variant in IOU:
            _iou = np.array(IOU[_variant])
            _row.update(median_iou=round(float(np.median(_iou[:, 0])), 3), median_iou_edge_tolerant=round(float(np.median(_iou[:, 1])), 3),
                        share_iou_below_070=round(float((_iou[:, 0] < 0.70).mean()), 3))
        RESULTS.append(_row)
    for _row in RESULTS:
        _row["vs_reference"] = round(_row["R@1"] - RESULTS[0]["R@1"], 1)
        _difference = metrics.paired_difference(HITS["small_single"].astype(float), HITS[_row["variant"]].astype(float))
        _row["vs_small_single"] = round(100 * _difference["difference"], 1)
        _row["vs_small_single_interval"] = f"{100 * _difference['low']:.1f} to {100 * _difference['high']:.1f}"

    _flags = {"clean": np.array([not q["occluded"] and not q["truncated"] for q in PHOTOS]),
              "truncated": np.array([bool(q["truncated"]) and not q["occluded"] for q in PHOTOS]),
              "occluded": np.array([bool(q["occluded"]) for q in PHOTOS])}
    BY_FLAG = [{"variant": v, **{f"R@1_{name}": round(100 * HITS[v][m].mean(), 1) for name, m in _flags.items()}} for v in VARIANTS]
    BY_FLAG.insert(0, {"variant": "photos", **{f"R@1_{name}": int(m.sum()) for name, m in _flags.items()}})

    chart, _axis = plt.subplots(figsize=(8, 3.4))
    _values = [r["R@1"] for r in RESULTS]
    _bounds = [tuple(float(x) for x in r["R@1_interval"].split(" to ")) for r in RESULTS]
    _axis.bar(VARIANTS, _values, color=["#4c72b0"] + ["#888888"] * (len(VARIANTS) - 1))
    _axis.errorbar(VARIANTS, _values, yerr=[[v - lo for v, (lo, _) in zip(_values, _bounds)], [hi - v for v, (_, hi) in zip(_values, _bounds)]],
                   fmt="none", ecolor="black", capsize=3)
    _axis.set_ylabel("R@1 (%)")
    _axis.set_ylim(min(_values) - 10, max(_values) + 5)
    _axis.tick_params(axis="x", labelrotation=30)
    chart.tight_layout()
    print("RESULTS", RESULTS)
    print("BY_FLAG", BY_FLAG)
    mo.vstack([mo.ui.table(RESULTS, selection=None), mo.md("**R@1 by Pix3D flag** (first row: number of photos)"),
               mo.ui.table(BY_FLAG, selection=None), chart])
    return BY_FLAG, RESULTS, chart


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Example query images

    One row per photo (the first 8 photos of the subset), one column per variant, in the order of the table
    at the top. This shows what the search actually sees: for example a chair without its cushion
    (`small_single`), the cushion back with SAM large, a thin ring of floor with the grown masks, or the
    whole room corner with `box_only`.
    """)
    return


@app.cell
def _(EXAMPLES, Image, ImageDraw, VARIANTS):
    _tile, _label = 112, 14
    example_sheet = Image.new("RGB", (len(VARIANTS) * _tile, _label + len(EXAMPLES) * (_tile + _label)), "white")
    _draw = ImageDraw.Draw(example_sheet)
    for _c, _variant in enumerate(VARIANTS):
        _draw.text((_c * _tile + 2, 1), _variant[:18], fill="black")
    for _r, (_query_id, _queries) in enumerate(EXAMPLES):
        _y = _label + _r * (_tile + _label)
        _draw.text((2, _y + 1), _query_id, fill="black")
        for _c, _variant in enumerate(VARIANTS):
            example_sheet.paste(Image.fromarray(_queries[_variant]).resize((_tile, _tile)), (_c * _tile, _y + _label))
    example_sheet
    return (example_sheet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Save the results

    The button writes the results into a new run folder, `runs/<date>_pix3d_<backbone>_mask_variants/`:
    `config.yaml` (all settings), `metrics.json` (the photos, timing, and all tables), `r1_chart.png`, and
    `example_sheet.png`. As a script, the results are saved without the button.
    """)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save the results")
    save_button
    return (save_button,)


@app.cell
def _(
    BY_FLAG,
    CFG,
    GROW_FRACTION,
    PHOTOS,
    RESULTS,
    SEED,
    TIMING,
    chart,
    datetime,
    example_sheet,
    extractor,
    git_state,
    json,
    make_run_name,
    mo,
    paths,
    save_button,
    save_config,
):
    mo.stop(not save_button.value and mo.app_meta().mode != "script", mo.md("Press the button to save the results."))
    _name = make_run_name("pix3d", extractor.spec.name, "mask_variants")
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
        "photo_set": "pix3d seen_meshes train + val (test excluded), random subset",
        "seed": SEED,
        "grow_fraction": GROW_FRACTION,
        "photos": [q["query_id"] for q in PHOTOS],
        "timing": TIMING,
        "results": RESULTS,
        "r1_by_flag": BY_FLAG,
    }
    (_run_dir / "metrics.json").write_text(json.dumps(_record, indent=2) + "\n")
    chart.savefig(_run_dir / "r1_chart.png", dpi=120)
    example_sheet.save(_run_dir / "example_sheet.png")
    print("SAVED", str(_run_dir.relative_to(paths.ROOT)))
    mo.md(f"Saved to `{_run_dir.relative_to(paths.ROOT)}`.")
    return


if __name__ == "__main__":
    app.run()
