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
    # 05: Segment Check

    This notebook checks the masks that SAM 2.1 makes for the Pix3D photos (`scripts/02_segment_photos.py`).
    SAM gets the box around each photo's reference mask (the mask that comes with Pix3D) and makes its
    own mask. The notebook then answers two questions:

    1. **How close are SAM's masks to the reference masks?** (IoU, sections 4 and 5). This is a
       diagnostic: it shows where and how SAM differs, with no pass mark.
    2. **How much retrieval do we lose with SAM's masks?** (section 6). Two R@1 values on the same photos:
       with the reference masks (the optimistic ceiling) and with SAM's masks (what the demo gets).

    **About the reference masks.** The Pix3D masks look like the outline of the aligned 3D mesh, not a
    trace of the visible pixels: they include parts hidden behind other objects, a blanket lying on a
    chair, and the glass top of a table. A few are a few pixels off the photo, which costs a lot of IoU on
    thin legs. So a low IoU is not always a SAM error; section 5 shows the cases.

    **Why there is no pass mark.** Because the reference masks carry information that is not in the
    photo (hidden parts, the exact outline of the mesh), no segmenter can reach them. An earlier version of
    this notebook had targets (median IoU 0.90, R@1 at most 3 points below the reference); SAM 2.1 small
    lost 10.6 points and SAM 2.1 large about 8, and a test with no mask at all lost only 12. So the
    reference R@1 is reported as the ceiling, the SAM R@1 as the honest number for the demo, and the
    gap is closed in training (training queries with SAM masks).

    **Which photos.** The tuning photos: the train and val parts of the `seen_meshes` split (8,557). The
    1,510 test photos are kept for the final end-to-end result, so that no setting is chosen by looking at them.

    **How to run**

    - Full outputs (GPU, about an hour with SAM 2.1 large):
      `python scripts/02_segment_photos.py --dataset pix3d --split seen_meshes --parts train val`.
      Mini outputs: press the button in section 2, or run the script with `--mini`.
    - Then, from the repository root: `marimo edit notebooks/05_segment_check.py` (interactively), or
      `python notebooks/05_segment_check.py` (as a script; it checks the full outputs).
    - Section 6 needs the GPU, the DINOv3 weights, and the Pix3D renders.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules, and reads all settings
    (`configs/*.yaml`), and sets:

    | Setting | Value | Meaning |
    |:---|:---|:---|
    | `GRID_PHOTOS` | 12 | Photos per overlay grid in section 5 |
    | `QUERY_CHUNK` | 512 | Query images per feature batch in section 6 |
    """)
    return


@app.cell
def _():
    import json
    import random
    import sys
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
    from src.config import load_config  # noqa: E402
    from src.data.catalog import read_table  # noqa: E402
    from src.models.backbones import FeatureExtractor, object_masks  # noqa: E402
    from src.pipeline import segment_photos  # noqa: E402
    from src.preprocess.queries import QueryJob, make_query_images  # noqa: E402
    from src.preprocess.segment import read_mask  # noqa: E402
    from src.retrieval import metrics, search  # noqa: E402

    CFG = load_config()
    BACKGROUND = tuple(CFG["data"]["background_rgb"])
    GRID_PHOTOS = 12
    QUERY_CHUNK = 512
    SEED = 0
    VIEWS = 25
    return (
        BACKGROUND,
        CFG,
        Counter,
        FeatureExtractor,
        GRID_PHOTOS,
        Image,
        ImageDraw,
        QUERY_CHUNK,
        QueryJob,
        SEED,
        VIEWS,
        json,
        make_query_images,
        metrics,
        np,
        object_masks,
        paths,
        plt,
        random,
        read_mask,
        read_table,
        search,
        segment_photos,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Mini run (optional) and the outputs to check

    The button runs the segment step on 3 photos per category (`segment_photos.run(..., mini=True)`,
    the same function as `scripts/02_segment_photos.py --dataset pix3d --mini`). It writes only to
    `test_outputs/segment_mini/` and takes about 10 seconds on the GPU (about a minute on the CPU).
    Then choose which outputs the rest of the notebook checks: `full` (the real outputs in `data/`)
    or `mini`. When the notebook runs as a script, the full outputs are checked.
    """)
    return


@app.cell
def _(mo):
    mini_button = mo.ui.run_button(label="Make the mini outputs")
    mini_button
    return (mini_button,)


@app.cell
def _(CFG, mini_button, segment_photos, torch):
    # Runs only when the button is pressed; otherwise mini_report is None and nothing is written.
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    mini_report = segment_photos.run("pix3d", CFG, mini=True, device=_device) if mini_button.value else None
    return (mini_report,)


@app.cell
def _(mini_report, mo, segment_photos):
    _ = mini_report  # makes this cell run again after a mini run, so that new outputs appear in the list
    OUTPUTS = {
        name: root
        for name, root in {"full": segment_photos.output_root(False), "mini": segment_photos.output_root(True)}.items()
        if segment_photos.results_file("pix3d", root).exists()
    }
    mo.stop(not OUTPUTS, mo.md("No segment outputs yet. Run the segment step, or press the button above."))
    outputs_choice = mo.ui.dropdown(list(OUTPUTS), value=next(iter(OUTPUTS)), label="outputs to check")
    outputs_choice
    return OUTPUTS, outputs_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Report and completeness

    The next cell reads the report and the results table of the step, and joins each row with its
    catalog row (photo, reference mask, mesh, and the Pix3D flags). The table also holds the rows of
    the test photos (made for the final result); the cell leaves them out, so that every value below
    comes from the tuning photos only.

    | Field | Meaning | Expected |
    |:---|:---|:---|
    | `photos` | Photos the last run was asked to segment | 8,557 (full), 27 (mini) |
    | `prompt` | Where SAM's box came from | reference_box |
    | `box_mask` | Which of SAM's masks is kept (`segment.segmenter.box_mask`) | single |
    | `status` | Rows per status over all runs (done, failed), test photos included | no failed |
    | `test_rows_left_out` | Rows of test photos that the checks below do not use | 1,510 (full); a few (mini: its photos come from all parts) |
    | `with_mask_file` | Tuning rows whose mask file exists | = photos |
    """)
    return


@app.cell
def _(Counter, OUTPUTS, json, mo, outputs_choice, paths, read_table, segment_photos):
    ROOT = OUTPUTS[outputs_choice.value]
    REPORT = json.loads(segment_photos.report_file("pix3d", ROOT).read_text())
    _catalog = {q["query_id"]: q for q in read_table(paths.QUERIES_CSV)}
    _results = read_table(segment_photos.results_file("pix3d", ROOT))
    _tuning = {item.photo_id for item in segment_photos.pix3d_items("seen_meshes", ["train", "val"])}
    # One record per tuning photo with an IoU: the results row, the catalog row, and the mask file.
    ROWS = []
    for _row in _results:
        if _row["iou"] in ("", None) or _row["photo_id"] not in _tuning:
            continue
        _query = _catalog[_row["photo_id"]]
        ROWS.append({
            **_query,
            "iou": float(_row["iou"]),
            "iou_edge_tolerant": float(_row["iou_edge_tolerant"]),
            "sam_score": float(_row["sam_score"]),
            "sam_mask_path": ROOT / "pix3d" / _query["category"] / (_query["query_id"].split("/")[-1] + ".png"),
        })
    _summary = {
        "photos": REPORT["photos"],
        "prompt": REPORT["prompt"],
        "box_mask": REPORT["box_mask"],
        "status": dict(Counter(r["status"] for r in _results)),
        "test_rows_left_out": sum(r["photo_id"] not in _tuning for r in _results),
        "with_mask_file": sum(r["sam_mask_path"].exists() for r in ROWS),
    }
    print("REPORT", _summary)
    mo.vstack([mo.ui.table([_summary], selection=None),
               mo.md(f"Failed photos: {REPORT['failed']}" if REPORT["failed"] else "No failed photos.")])
    return REPORT, ROOT, ROWS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Agreement with the reference masks

    **IoU** = (pixels that both masks call object) / (pixels that either mask calls object); 1.0 means
    identical. **Edge-tolerant IoU** ignores the pixels near the edge of the reference mask (0.5 % of the
    longer photo side: 3 pixels at 640 pixels, `segment.check.edge_band_fraction`), so that a reference
    mask that is a few pixels off, or a slightly different edge, does not count as an error.

    | IoU | What it usually means |
    |:---|:---|
    | 0.95 or more | Only edge differences |
    | 0.90 to 0.95 | Small misses: a thin leg, a gap between slats |
    | 0.70 to 0.90 | A visible part differs: a missing cushion, the floor included, a hidden part in the reference |
    | below 0.70 | Another object or only a part: only the frame, two objects merged |

    The first table summarizes both measures. The chart shows how the IoU
    values spread over the photos. The two tables below give the median per category and per Pix3D flag
    (`truncated`: cut by the photo border; `occluded`: partly hidden by another object). Thin objects
    (chairs, tools) have more edge for their area, so their IoU is lower even when the mask looks right.
    """)
    return


@app.cell
def _(ROWS, mo, np, plt):
    _table = []
    for _column in ("iou", "iou_edge_tolerant"):
        _values = np.array([r[_column] for r in ROWS])
        _median, _below = float(np.median(_values)), float((_values < 0.70).mean())
        _table.append({
            "measure": _column,
            "photos": len(_values),
            "median": round(_median, 3),
            "mean": round(float(_values.mean()), 3),
            "share_below_0.70": round(_below, 3),
            "share_below_0.90": round(float((_values < 0.90).mean()), 3),
        })
    print("AGREEMENT", _table)
    iou_chart, _axis = plt.subplots(figsize=(7, 3))
    _bins = np.linspace(0, 1, 41)
    _axis.hist([r["iou"] for r in ROWS], bins=_bins, alpha=0.6, label="IoU")
    _axis.hist([r["iou_edge_tolerant"] for r in ROWS], bins=_bins, alpha=0.6, label="edge-tolerant IoU")
    _axis.axvline(0.70, color="grey", linestyle=":", linewidth=1)
    _axis.set_xlabel("IoU with the reference mask")
    _axis.set_ylabel("photos")
    _axis.legend()
    iou_chart.tight_layout()
    mo.vstack([mo.ui.table(_table, selection=None), iou_chart])
    return


@app.cell
def _(ROWS, mo, np):
    def _breakdown(key_of):
        groups = {}
        for row in ROWS:
            groups.setdefault(key_of(row), []).append(row)
        return [
            {
                "group": key,
                "photos": len(rows),
                "median_iou": round(float(np.median([r["iou"] for r in rows])), 3),
                "median_edge_tolerant": round(float(np.median([r["iou_edge_tolerant"] for r in rows])), 3),
                "share_below_0.70": round(float(np.mean([r["iou"] < 0.70 for r in rows])), 3),
            }
            for key, rows in sorted(groups.items())
        ]

    _by_category = _breakdown(lambda r: r["category"])
    _by_flag = _breakdown(lambda r: "occluded" if r["occluded"] else ("truncated" if r["truncated"] else "clean"))
    print("BY_CATEGORY", _by_category)
    print("BY_FLAG", _by_flag)
    mo.vstack([mo.md("**Per category**"), mo.ui.table(_by_category, selection=None),
               mo.md("**Per Pix3D flag** (occluded wins over truncated)"), mo.ui.table(_by_flag, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Overlays

    Each tile shows a photo with both masks: **green** = both masks call it object; **blue** = only the
    reference mask (Pix3D); **red** = only SAM. The yellow box is SAM's prompt. The caption gives the
    query id and both IoU values.

    Typical patterns: thin blue and red lines on opposite sides of a leg = the reference mask is shifted;
    a large blue area = SAM left out a part (a cushion, a hidden part, a blanket); a large red area = SAM
    added something (the floor, a neighbouring object).

    Choose the order (the lowest IoU, photos around the median, or random photos) and a category.
    """)
    return


@app.cell
def _(ROWS, mo):
    order_choice = mo.ui.dropdown(["lowest IoU", "around the median", "random"], value="lowest IoU", label="photos")
    category_choice = mo.ui.dropdown(["all"] + sorted({r["category"] for r in ROWS}), value="all", label="category")
    mo.hstack([order_choice, category_choice], justify="start")
    return category_choice, order_choice


@app.cell
def _(GRID_PHOTOS, Image, ImageDraw, ROWS, SEED, category_choice, mo, np, order_choice, paths, random, read_mask):
    _rows = [r for r in ROWS if category_choice.value in ("all", r["category"])]
    _rows = sorted(_rows, key=lambda r: r["iou"])
    if order_choice.value == "around the median":
        _middle = len(_rows) // 2
        _rows = _rows[max(0, _middle - GRID_PHOTOS // 2) : _middle + GRID_PHOTOS // 2]
    elif order_choice.value == "random":
        _rows = random.Random(SEED).sample(_rows, min(GRID_PHOTOS, len(_rows)))
    _rows = _rows[:GRID_PHOTOS]

    def overlay(row, height=260):
        """The photo with both masks coloured (see the legend above), scaled to `height` pixels."""
        with Image.open(paths.resolve(row["photo_path"])) as photo:
            pixels = np.asarray(photo.convert("RGB")).astype(float)
        reference, sam = read_mask(row["mask_path"]), read_mask(row["sam_mask_path"])
        pixels[reference & sam] = pixels[reference & sam] * 0.5 + np.array([0, 255, 0]) * 0.5
        pixels[reference & ~sam] = (0, 120, 255)
        pixels[sam & ~reference] = (255, 40, 0)
        tile = Image.fromarray(pixels.astype(np.uint8))
        ys, xs = np.nonzero(reference)
        ImageDraw.Draw(tile).rectangle((xs.min(), ys.min(), xs.max(), ys.max()), outline=(255, 220, 0), width=max(2, tile.width // 200))
        tile = tile.resize((max(1, round(tile.width * height / tile.height)), height))
        return mo.vstack([mo.image(tile), mo.md(f"`{row['query_id']}` IoU {row['iou']:.2f} / {row['iou_edge_tolerant']:.2f}")])

    mo.hstack([overlay(r) for r in _rows], wrap=True, justify="start") if _rows else mo.md("No photos in this selection.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Retrieval with SAM masks against retrieval with the reference masks

    For every photo of section 4, the next cells make a second query image: the same preparation as the
    normal query image (grey background, tight square crop, 224 × 224), but cut out with SAM's mask. The
    query images are saved in `<masks>/pix3d_sam_queries/` and made again on every run (about 2 minutes),
    so that they always match the current masks. Both sets of query images are
    then searched against the 395 Pix3D meshes with the frozen backbone (zero-shot, view score), exactly
    like the lighting test. So the only difference between the two results is the mask.

    | Field | Meaning |
    |:---|:---|
    | `masks` | reference (Pix3D; the optimistic ceiling) or SAM (what the demo gets) |
    | `R@1`, `R@5` | Share of photos whose correct mesh is ranked first, or among the first 5 (%) |
    | `R@1_interval` | 95 % bootstrap interval of R@1 |
    | `vs_reference` | R@1 with SAM masks minus R@1 with reference masks, in points (the gap that training should close) |
    | `vs_reference_interval` | 95 % interval of that difference (paired: the same photos) |

    The second table splits the photos by their IoU: if retrieval drops mostly for low-IoU photos, the
    masks are the cause; if it drops evenly, the cause is elsewhere. The first run takes about 3 minutes
    (query images, then features of 9,875 renders and of both query sets); the button starts it.
    """)
    return


@app.cell
def _(mo):
    retrieval_button = mo.ui.run_button(label="Compare retrieval with SAM masks and reference masks")
    retrieval_button
    return (retrieval_button,)


@app.cell
def _(BACKGROUND, CFG, QueryJob, ROOT, ROWS, make_query_images, mo, retrieval_button):
    mo.stop(not retrieval_button.value and mo.app_meta().mode != "script", mo.md("Press the button to run section 6."))
    # Query images cut out with SAM's masks (the same function as the query step), made again every run.
    _jobs = [
        QueryJob(
            query_id=r["query_id"],
            photo_path=r["photo_path"],
            mask_path=str(r["sam_mask_path"]),
            output_path=str(ROOT / "pix3d_sam_queries" / r["category"] / (r["query_id"].split("/")[-1] + ".png")),
            background_rgb=BACKGROUND,
            margin=CFG["data"]["crop_margin"],
            output_size=CFG["data"]["image_size"],
            overwrite=True,  # always from the current masks
        )
        for r in ROWS
    ]
    _results = make_query_images(_jobs)  # one result per job, in the same order
    SAM_QUERY_PATHS = {job.query_id: job.output_path for job, result in zip(_jobs, _results) if result["status"] != "failed"}
    print("SAM_QUERIES", {"made_or_found": len(SAM_QUERY_PATHS), "failed": len(_jobs) - len(SAM_QUERY_PATHS)})
    return (SAM_QUERY_PATHS,)


@app.cell
def _(BACKGROUND, CFG, FeatureExtractor, Image, QUERY_CHUNK, ROWS, SAM_QUERY_PATHS, VIEWS, np, object_masks, paths, read_table, tqdm):
    extractor = FeatureExtractor(CFG["backbone"])

    def features_of(files):
        """Feature vectors of a list of 224 x 224 images (grey background), in chunks."""
        parts = []
        for start in tqdm(range(0, len(files), QUERY_CHUNK), desc="features", unit="chunk"):
            images = np.stack([np.asarray(Image.open(paths.resolve(f)).convert("RGB")) for f in files[start : start + QUERY_CHUNK]])
            parts.append(extractor.extract(images, object_masks(images, BACKGROUND)))
        return np.concatenate(parts)

    COMPARED = [r for r in ROWS if r["query_id"] in SAM_QUERY_PATHS]  # photos with both query images
    GALLERY = [m for m in read_table(paths.MESHES_CSV) if m["dataset"] == "pix3d"]
    _view_files = [f"{m['render_dir']}/view_{v:02d}.png" for m in GALLERY for v in range(VIEWS)]
    VIEW_FEATURES = features_of(_view_files).reshape(len(GALLERY), VIEWS, -1)
    REFERENCE_FEATURES = features_of([r["query_path"] for r in COMPARED])
    SAM_FEATURES = features_of([SAM_QUERY_PATHS[r["query_id"]] for r in COMPARED])
    print("FEATURES", {"gallery": VIEW_FEATURES.shape, "queries": REFERENCE_FEATURES.shape})
    return COMPARED, GALLERY, REFERENCE_FEATURES, SAM_FEATURES, VIEW_FEATURES


@app.cell
def _(COMPARED, GALLERY, REFERENCE_FEATURES, SAM_FEATURES, VIEW_FEATURES, metrics, mo, np, search):
    _column = {m["mesh_id"]: i for i, m in enumerate(GALLERY)}
    _correct = np.array([_column[r["mesh_id"]] for r in COMPARED])
    hits = {}
    _table = []
    for _name, _features in (("reference", REFERENCE_FEATURES), ("SAM", SAM_FEATURES)):
        _ranks = metrics.correct_ranks(search.view_scores(_features, VIEW_FEATURES), _correct)
        hits[_name] = _ranks <= 1
        _low, _high = metrics.bootstrap_interval(hits[_name])
        _table.append({"masks": _name, "photos": len(_ranks), "R@1": round(100 * metrics.recall_at_k(_ranks, 1), 1),
                       "R@1_interval": f"{100 * _low:.1f} to {100 * _high:.1f}", "R@5": round(100 * metrics.recall_at_k(_ranks, 5), 1)})
    _difference = metrics.paired_difference(hits["reference"].astype(float), hits["SAM"].astype(float))
    _table[1]["vs_reference"] = round(100 * _difference["difference"], 1)
    _table[1]["vs_reference_interval"] = f"{100 * _difference['low']:.1f} to {100 * _difference['high']:.1f}"

    _iou = np.array([r["iou"] for r in COMPARED])
    _by_iou = []
    for _low_edge, _high_edge in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        _in = (_iou >= _low_edge) & (_iou < _high_edge)
        if _in.any():
            _by_iou.append({"iou_range": f"{_low_edge:.1f} to {min(_high_edge, 1.0):.1f}", "photos": int(_in.sum()),
                            "R@1_reference": round(100 * hits["reference"][_in].mean(), 1),
                            "R@1_SAM": round(100 * hits["SAM"][_in].mean(), 1)})
    print("RETRIEVAL", _table)
    print("BY_IOU", _by_iou)
    mo.vstack([mo.ui.table(_table, selection=None), mo.md("**By IoU of the SAM mask**"), mo.ui.table(_by_iou, selection=None)])
    return


if __name__ == "__main__":
    app.run()
