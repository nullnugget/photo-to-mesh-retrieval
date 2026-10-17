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
    # 01: Query Check

    This notebook checks the Pix3D query images that `scripts/01_prepare_pix3d.py` made.
    A query image is a photo after preparation: the background is replaced with neutral grey,
    the image is cropped tightly around the object, padded to a square, and resized to 224 × 224.

    The notebook shows:

    1. the catalog: how many queries and shapes exist, per category, and which photos were left out;
    2. a contact sheet: random queries from every category, to check them by eye;
    3. examples of the preparation steps: photo, mask, and query side by side;
    4. automatic checks over all queries: size, framing, and background.

    **How to run**

    - Run `python scripts/01_prepare_pix3d.py` first.
    - Then, from the repository root: `python notebooks/01_query_check.py` (as a script), or
      `marimo edit notebooks/01_query_check.py` (interactively).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules, and reads the data settings
    (image size and background colour) from `configs/data.yaml`. It also fixes the random seed
    for the samples, so that the notebook shows the same images every time.
    """)
    return


@app.cell
def _():
    import json
    import random
    import sys
    from collections import Counter
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw

    # The notebook is in <repository>/notebooks/, so the repository root is one level up.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))  # makes the project package "src" importable
    from src import paths  # noqa: E402
    from src.config import load_config  # noqa: E402
    from src.data.catalog import read_table  # noqa: E402
    from src.data.pix3d import read_photo_and_mask  # noqa: E402

    DATA_CFG = load_config(["data.yaml"])["data"]
    IMAGE_SIZE = DATA_CFG["image_size"]  # side length of every query (224)
    BACKGROUND = np.array(DATA_CFG["background_rgb"], dtype=np.uint8)  # the neutral grey
    SAMPLE_SEED = 0  # fixed seed for the random samples below
    return (
        BACKGROUND,
        Counter,
        IMAGE_SIZE,
        Image,
        ImageDraw,
        json,
        np,
        paths,
        random,
        read_photo_and_mask,
        read_table,
        SAMPLE_SEED,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Catalog and preparation report

    The next cell reads the query and shape tables of the catalog and the preparation report.
    It shows the number of queries and shapes per category, and the photos that the preparation
    left out, with the reason.
    """)
    return


@app.cell
def _(Counter, json, mo, paths, read_table):
    queries = [q for q in read_table(paths.QUERIES_CSV) if q["dataset"] == "pix3d"]
    shapes = [s for s in read_table(paths.SHAPES_CSV) if s["dataset"] == "pix3d"]
    report = json.loads((paths.CATALOG_DIR / "pix3d_report.json").read_text())

    _queries_per_category = Counter(q["category"] for q in queries)
    _shapes_per_category = Counter(s["category"] for s in shapes)
    category_rows = [
        {"category": c, "queries": _queries_per_category[c], "shapes": _shapes_per_category[c]}
        for c in sorted(_queries_per_category)
    ]
    print("CATALOG", {"queries": len(queries), "shapes": len(shapes), "left_out": [r["query_id"] for r in report["left_out"]]})
    mo.vstack(
        [
            mo.md(f"**{len(queries)} queries, {len(shapes)} shapes.** Left out: "
                  + ", ".join(f"`{r['query_id']}` ({r['message'].split(':')[0]})" for r in report["left_out"])),
            mo.ui.table(category_rows, selection=None),
        ]
    )
    return (queries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Contact sheet

    The next cell shows 8 random queries from each category, one category per row.

    What to look for:

    - the background is a uniform grey, with no remains of the original background;
    - the object fills the frame along its longer side, and is centred along its shorter side;
    - the object is complete (not cut by the crop). Photos where the object is cut by the photo
      border itself ("truncated" in Pix3D) stay cut.
    """)
    return


@app.cell
def _(Image, ImageDraw, IMAGE_SIZE, paths, queries, random, SAMPLE_SEED):
    def contact_sheet(rows_of_paths, labels, tile=112, label_width=90):
        """Draw a grid: one row per label, one tile per image path."""
        columns = max(len(row) for row in rows_of_paths)
        sheet = Image.new("RGB", (label_width + columns * tile, len(rows_of_paths) * tile), "white")
        draw = ImageDraw.Draw(sheet)
        for r, (row, label) in enumerate(zip(rows_of_paths, labels)):
            draw.text((4, r * tile + tile // 2 - 6), label, fill="black")
            for c, image_path in enumerate(row):
                with Image.open(image_path) as image:
                    sheet.paste(image.resize((tile, tile)), (label_width + c * tile, r * tile))
        return sheet

    _rng = random.Random(SAMPLE_SEED)
    _categories = sorted({q["category"] for q in queries})
    _rows = []
    for _category in _categories:
        _pool = [q for q in queries if q["category"] == _category]
        _rows.append([paths.resolve(q["query_path"]) for q in _rng.sample(_pool, min(8, len(_pool)))])
    query_sheet = contact_sheet(_rows, _categories)
    print("CONTACT_SHEET", {"size": query_sheet.size, "image_size": IMAGE_SIZE})
    query_sheet
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Preparation steps

    The next cell shows, for a few photos, the original photo, its mask (white = object), and the
    resulting query. The examples include a photo with several products (the mask selects one),
    a truncated photo, and a photo with an occluding object.
    """)
    return


@app.cell
def _(Image, mo, np, paths, queries, read_photo_and_mask):
    _examples = ["pix3d/chair/0001", "pix3d/bookcase/0001", "pix3d/table/0046"]
    _truncated = next(q["query_id"] for q in queries if q["truncated"])
    _occluded = next(q["query_id"] for q in queries if q["occluded"])
    _by_id = {q["query_id"]: q for q in queries}
    _cells = []
    for _query_id in _examples + [_truncated, _occluded]:
        _q = _by_id[_query_id]
        _photo, _mask = read_photo_and_mask(_q["photo_path"], _q["mask_path"])
        _row = [
            mo.image(Image.fromarray(_photo).resize((224, int(224 * _photo.shape[0] / _photo.shape[1])))),
            mo.image(Image.fromarray(_mask.astype(np.uint8) * 255).resize((224, int(224 * _mask.shape[0] / _mask.shape[1])))),
            mo.image(Image.open(paths.resolve(_q["query_path"]))),
        ]
        _cells.append(mo.vstack([mo.md(f"`{_query_id}`"), mo.hstack(_row, justify="start")]))
    mo.vstack(_cells)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Automatic checks over all queries

    The next cell opens every query and measures:

    | Measure | Meaning | Expected |
    |:---|:---|:---|
    | `wrong_size` | Number of queries that are not 224 × 224 RGB | 0 |
    | `fill` | Longer side of the object (the non-grey pixels) divided by 224. A tight crop gives 1.0 | 1.0 for almost all queries |
    | `fill_below_0.95` | Number of queries whose object fills less than 95 % of the longer side | close to 0 |
    | `grey_fraction` | Share of pixels that are exactly the background grey (the background plus the padding) | depends on the object shape |
    | `corners_grey` | Share of queries whose 4 corner pixels are all grey | high (an object that fills a corner, for example a square table seen from above, is possible) |

    The object pixels are found as the pixels that differ from the background grey. An object pixel
    that happens to have exactly this grey is counted as background; this is rare and has no
    visible effect on the measures. This cell reads 10,000 images, so it takes about half a minute.
    """)
    return


@app.cell
def _(BACKGROUND, Image, IMAGE_SIZE, mo, np, paths, queries):
    _fills, _grey_fractions, _corners_grey, _wrong_size = [], [], 0, 0
    for _q in queries:
        _query = np.asarray(Image.open(paths.resolve(_q["query_path"])))
        if _query.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            _wrong_size += 1
            continue
        _is_grey = (_query == BACKGROUND).all(axis=2)  # True where the pixel is exactly the background grey
        _object = ~_is_grey
        _rows = np.flatnonzero(_object.any(axis=1))
        _cols = np.flatnonzero(_object.any(axis=0))
        _fills.append(max(_rows[-1] - _rows[0] + 1, _cols[-1] - _cols[0] + 1) / IMAGE_SIZE if _rows.size else 0.0)
        _grey_fractions.append(_is_grey.mean())
        _corners_grey += int(_is_grey[0, 0] and _is_grey[0, -1] and _is_grey[-1, 0] and _is_grey[-1, -1])
    _fills = np.array(_fills)
    checks = {
        "queries": len(queries),
        "wrong_size": _wrong_size,
        "fill_median": round(float(np.median(_fills)), 3),
        "fill_min": round(float(_fills.min()), 3),
        "fill_below_0.95": int((_fills < 0.95).sum()),
        "grey_fraction_median": round(float(np.median(_grey_fractions)), 3),
        "corners_grey": round(_corners_grey / len(_fills), 3),
    }
    print("CHECKS", checks)
    mo.ui.table([checks], selection=None)
    return


if __name__ == "__main__":
    app.run()
