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
    # 02: Render Check

    This notebook checks the renders that the render step made (`scripts/04_render.py`): 25 grey
    views of every gallery mesh, cropped like the query images.

    It shows:

    1. optionally, a mini run of the render step (2 meshes per category), to check the step quickly;
    2. completeness: every mesh has its 25 views; the failures and the clipping count of the run;
    3. automatic checks over all views: size, framing, background, and separation from the background;
    4. a contact sheet: one mesh per category, a query photo next to its 25 views;
    5. the 25 views of any mesh, in large size.

    **How to run**

    - Full renders: run `python scripts/04_render.py --dataset pix3d` first. Mini renders: press the
      button in section 2 (in the editor), or run the script with `--mini`.
    - Then, from the repository root: `marimo edit notebooks/02_render_check.py` (interactively), or
      `python notebooks/02_render_check.py -- --dataset abo` (as a script; it checks the full renders if
      they exist, otherwise the mini renders; without `--dataset` it checks Pix3D).
    - The checks over all views take about half a minute for Pix3D (9,875 views) and about 6 minutes for
      ABO (109,825 views).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules, including the render step, and
    sets the seed of the random samples. The cell after it chooses the dataset to check (`pix3d` or
    `abo`); as a script, the dataset comes from `--dataset`.
    """)
    return


@app.cell
def _():
    import json
    import random
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw

    # The notebook is in <repository>/notebooks/, so the repository root is one level up.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes the project package "src" importable
    from src import paths  # noqa: E402
    from src.config import load_config, load_yaml  # noqa: E402
    from src.data.catalog import read_table  # noqa: E402
    from src.pipeline import render_meshes  # noqa: E402

    CFG = load_config()
    BACKGROUND = np.array(CFG["render"]["background_rgb"], dtype=np.uint8)
    VIEWS = 25
    SAMPLE_SEED = 0
    return (
        BACKGROUND,
        CFG,
        Image,
        ImageDraw,
        SAMPLE_SEED,
        VIEWS,
        json,
        load_config,
        load_yaml,
        np,
        paths,
        random,
        read_table,
        render_meshes,
    )


@app.cell
def _(mo):
    # In the editor, choose here. As a script: python notebooks/02_render_check.py -- --dataset abo
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
    ## 2. Mini run (optional) and the renders to check

    The button runs the render step on 2 meshes per category (`render_meshes.run(..., mini=True)`,
    the same function as `scripts/04_render.py --mini`). It writes only to `test_outputs/render_mini/`
    and takes about 15 seconds. Then choose which renders the rest of the notebook checks: `full`
    (`data/derived/renders/`) or `mini`. When the notebook runs as a script, the button is not pressed.
    """)
    return


@app.cell
def _(mo):
    mini_button = mo.ui.run_button(label="Make the mini renders")
    mini_button
    return (mini_button,)


@app.cell
def _(DATASET, load_config, mini_button, render_meshes):
    # Runs only when the button is pressed; otherwise mini_report is None and nothing is rendered.
    mini_report = render_meshes.run(DATASET, load_config(), mini=True) if mini_button.value else None
    return (mini_report,)


@app.cell
def _(DATASET, mini_report, mo, render_meshes):
    _ = mini_report  # makes this cell run again after a mini run, so that new renders appear in the list
    ROOTS = {
        name: root
        for name, root in {"full": render_meshes.output_root(False), "mini": render_meshes.output_root(True)}.items()
        if render_meshes.render_report_file(DATASET, root).exists()
    }
    mo.stop(not ROOTS, mo.md("No renders yet. Press the button for a mini run, or run `scripts/04_render.py`."))
    root_choice = mo.ui.dropdown(list(ROOTS), value=next(iter(ROOTS)), label="renders to check")
    root_choice
    return ROOTS, root_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Completeness and the run report

    The next cell compares the render folders with the catalog, and reads the report and the
    settings record. The report adds up all runs (a stopped and restarted render is one result).

    | Field | Meaning | Expected |
    |:---|:---|:---|
    | `meshes_in_run` | Meshes that the runs had to render (all catalog meshes; 2 per category in a mini run) | |
    | `complete` | Meshes whose folder has all 25 views | `meshes_in_run` |
    | `incomplete` | Folders with fewer views, or `.part` folders left by an interrupted run | 0 |
    | `failed` | Meshes that could not be rendered (the reasons are listed below the table) | 0 |
    | `views_touching_border` | Views in which the object touched the border of the raw render (clipping) | 0 |
    | `overflow_retries` | View batches that were rasterized again because a rasterizer tile overflowed (repaired; for information) | small |
    | `light_setup` | The light setup of the renders | L2_dark |
    """)
    return


@app.cell
def _(DATASET, ROOTS, VIEWS, json, load_yaml, mo, paths, read_table, render_meshes, root_choice):
    root = ROOTS[root_choice.value]
    report = json.loads(render_meshes.render_report_file(DATASET, root).read_text())
    settings = load_yaml(root / paths.RENDER_CONFIG_FILE.name)["render"]
    _catalog = [m for m in read_table(paths.MESHES_CSV) if m["dataset"] == DATASET]
    rendered = [m for m in _catalog if render_meshes.render_folder(m, root).exists()]
    _incomplete = [m["mesh_id"] for m in rendered if len(list(render_meshes.render_folder(m, root).glob("view_*.png"))) != VIEWS]
    _incomplete += [str(p.relative_to(root)) for p in root.rglob("*.part")]
    _summary = {
        "meshes_in_run": report["meshes"],
        "complete": len(rendered) - len(_incomplete),
        "incomplete": len(_incomplete),
        "failed": len(report["failed"]),
        "views_touching_border": report["views_touching_border"],
        "overflow_retries": report["overflow_retries"],
        "light_setup": settings["lighting"],
    }
    print("COMPLETENESS", _summary)
    _notes = "".join(f"\n- failed: `{f['mesh_id']}` ({f['message']})" for f in report["failed"])
    _notes += "".join(f"\n- incomplete: `{i}`" for i in _incomplete)
    mo.vstack([
        mo.ui.table([_summary], selection=None),
        mo.md(f"Settings: rig `{settings['rig']}`, field of view {settings['fov_deg']}°, render size {settings['render_size']}, "
              f"output size {settings['output_size']}, normals `{settings['material']['normals']}`." + _notes),
    ])
    return rendered, root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Automatic checks over all views

    The next cell opens every view and measures:

    | Measure | Meaning | Expected |
    |:---|:---|:---|
    | `views` | Views checked | 25 per mesh |
    | `wrong_size` | Views that are not 224 x 224 RGB | 0 |
    | `fill_median`, `fill_min` | Longer side of the object (the non-background pixels) divided by 224; a tight crop gives 1.0 | about 1.0 |
    | `corners_background` | Share of views whose 4 corner pixels are the background grey | high (an object can fill a corner) |
    | `near_background_percent` | Share of object pixels within 8 grey levels of the background, where the object and the background look the same | about 1 % with L2_dark |
    | `mean_brightness` | Average grey value of the object pixels (0 to 255) | about 68 with L2_dark |

    Object pixels are found as the pixels that differ from the background grey. The full Pix3D
    renders are about 10,000 views, so this cell takes about half a minute.
    """)
    return


@app.cell
def _(BACKGROUND, Image, VIEWS, mo, np, render_meshes, rendered, root):
    _fills, _near, _brightness, _corners, _wrong, _count = [], [], [], 0, 0, 0
    for _mesh in rendered:
        _folder = render_meshes.render_folder(_mesh, root)
        for _v in range(VIEWS):
            _view = np.asarray(Image.open(_folder / render_meshes.VIEW_NAME.format(_v)))
            _count += 1
            if _view.shape != (224, 224, 3):
                _wrong += 1
                continue
            _object = (_view != BACKGROUND).any(axis=2)
            _rows, _cols = np.flatnonzero(_object.any(axis=1)), np.flatnonzero(_object.any(axis=0))
            _fills.append(max(_rows[-1] - _rows[0] + 1, _cols[-1] - _cols[0] + 1) / 224 if _rows.size else 0.0)
            _grey = _view.astype(np.float32).mean(axis=2)[_object]
            _near.append(np.abs(_grey - float(BACKGROUND[0])) <= 8)
            _brightness.append(_grey)
            _corners += int(not (_object[0, 0] or _object[0, -1] or _object[-1, 0] or _object[-1, -1]))
    _near, _brightness = np.concatenate(_near), np.concatenate(_brightness)
    view_checks = {
        "views": _count,
        "wrong_size": _wrong,
        "fill_median": round(float(np.median(_fills)), 3),
        "fill_min": round(float(np.min(_fills)), 3),
        "corners_background": round(_corners / max(len(_fills), 1), 3),
        "near_background_percent": round(float(_near.mean() * 100), 2),
        "mean_brightness": round(float(_brightness.mean()), 1),
    }
    print("VIEW_CHECKS", view_checks)
    mo.ui.table([view_checks], selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Contact sheet

    The next cell shows one random mesh per category, one row each: a query photo of the mesh, then
    its 25 views in rig order: the low ring (−15°, views 0–7), the middle ring (+15°, views 8–15),
    the high ring (+45°, views 16–23), and the top view (+85°, view 24). Each ring starts at the
    front of the mesh and goes around to the right.

    What to look for: the object is complete in every view, stands upright, is centred and fills
    the frame, is clearly darker than the background, and shows its 3D form through the shading.
    View 0 should show the front, like most photos.
    """)
    return


@app.cell
def _(DATASET, Image, ImageDraw, SAMPLE_SEED, VIEWS, paths, random, read_table, render_meshes, rendered, root):
    _rng = random.Random(SAMPLE_SEED)
    _photos = {}
    for _q in read_table(paths.QUERIES_CSV):
        if _q["dataset"] == DATASET:
            _photos.setdefault(_q["mesh_id"], _q)
    _tile, _label = 64, 16
    _sample = [_rng.choice([m for m in rendered if m["category"] == c]) for c in sorted({m["category"] for m in rendered})]
    contact_sheet = Image.new("RGB", ((1 + VIEWS) * _tile, len(_sample) * (_tile + _label)), "white")
    _draw = ImageDraw.Draw(contact_sheet)
    for _r, _mesh in enumerate(_sample):
        _y = _r * (_tile + _label)
        _draw.text((2, _y + 2), _mesh["mesh_id"], fill="black")
        if _mesh["mesh_id"] in _photos:
            with Image.open(paths.resolve(_photos[_mesh["mesh_id"]]["query_path"])) as _photo:
                contact_sheet.paste(_photo.resize((_tile, _tile)), (0, _y + _label))
        _folder = render_meshes.render_folder(_mesh, root)
        for _v in range(VIEWS):
            with Image.open(_folder / render_meshes.VIEW_NAME.format(_v)) as _view:
                contact_sheet.paste(_view.resize((_tile, _tile)), ((1 + _v) * _tile, _y + _label))
    print("CONTACT_SHEET", {"meshes": [m["mesh_id"] for m in _sample]})
    contact_sheet
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. All views of one mesh

    Choose a mesh in the list; the next cell shows its 25 views in a 5 x 5 grid (rows in rig order,
    as in section 5).
    """)
    return


@app.cell
def _(mo, rendered):
    mesh_choice = mo.ui.dropdown([m["mesh_id"] for m in rendered], value=rendered[0]["mesh_id"], label="mesh")
    mesh_choice
    return (mesh_choice,)


@app.cell
def _(Image, VIEWS, mesh_choice, render_meshes, rendered, root):
    _mesh = next(m for m in rendered if m["mesh_id"] == mesh_choice.value)
    _folder = render_meshes.render_folder(_mesh, root)
    _tile = 160
    mesh_grid = Image.new("RGB", (5 * _tile, 5 * _tile), "white")
    for _v in range(VIEWS):
        with Image.open(_folder / render_meshes.VIEW_NAME.format(_v)) as _view:
            mesh_grid.paste(_view.resize((_tile, _tile)), ((_v % 5) * _tile, (_v // 5) * _tile))
    mesh_grid
    return


if __name__ == "__main__":
    app.run()
