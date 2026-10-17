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
    # 03: Make queries

    This step makes the query images of one dataset, its rows in the catalog, and its split files.
    A query image is a photo after preparation: the background is replaced with neutral grey,
    the image is cropped tightly around the object, padded to a square, and resized to 224 × 224.

    | Output | Location |
    |:---|:---|
    | Query images | `data/derived/queries/<dataset>/<category>/<name>.png` |
    | Catalog tables (the rows of this dataset are replaced) | `data/derived/catalog/queries.csv`, `meshes.csv` |
    | Report: counts, left-out photos, settings | `data/derived/catalog/<dataset>_queries_report.json` |
    | Split files | `data/splits/<dataset>/seen_meshes.json`, `unseen_meshes.json`, `unseen_categories.json` |

    A **mini** run prepares only 5 photos per category and writes everything to
    `test_outputs/make_queries_mini/`, so that the real outputs are not touched.

    **How to run** (from the repository root)

    - In the editor: `marimo edit scripts/03_make_queries.py`. Set the options below, then press **Run**.
    - In the terminal: `python scripts/03_make_queries.py --dataset pix3d` (all photos, about 10 minutes),
      or add `--mini` (quick check, a few seconds). `--help` lists all options.

    The logic of the step is in `src/pipeline/make_queries.py`. This file only collects the options
    and starts the step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports

    The next cell makes the project package `src` importable and imports the step.
    """)
    return


@app.cell
def _():
    import argparse
    import sys
    from pathlib import Path

    # This file is in <repository>/scripts/, so the repository root is one level up.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes the project package "src" importable
    from src.config import add_config_args, load_config  # noqa: E402
    from src.pipeline import make_queries  # noqa: E402
    from src.preprocess.queries import default_workers  # noqa: E402

    return add_config_args, argparse, default_workers, load_config, make_queries, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Options

    | Option | Terminal | Meaning |
    |:---|:---|:---|
    | dataset | `--dataset pix3d` | The dataset to prepare. Only Pix3D is available now. |
    | mini | `--mini` | Prepare 5 photos per category, into `test_outputs/make_queries_mini/`. |
    | overwrite | `--overwrite` | Make the query images again, also when the files exist. Without it, existing images are kept, and only the catalog rows and the split files are written again. |
    | workers | `--workers 8` | Number of parallel processes that make the images. |
    | setting overrides | `--set data.crop_margin=0.05` | Change a value of `configs/*.yaml` for this run only. In the editor: one `KEY=VALUE` per line. |

    From the terminal, the options come from the command line. In the editor, they start at
    their default values.
    """)
    return


@app.cell
def _(add_config_args, argparse, default_workers, make_queries, mo, sys):
    _parser = argparse.ArgumentParser(
        prog="scripts/03_make_queries.py",
        description="Make the query images, catalog rows, and split files of one dataset.",
    )
    _parser.add_argument("--dataset", choices=make_queries.DATASETS, default="pix3d", help="dataset to prepare")
    _parser.add_argument("--mini", action="store_true", help="5 photos per category, into test_outputs/")
    _parser.add_argument("--overwrite", action="store_true", help="make the query images again if they exist")
    _parser.add_argument("--workers", type=int, default=default_workers(), help="parallel processes")
    add_config_args(_parser)
    is_script = mo.app_meta().mode == "script"  # True when started with "python scripts/03_make_queries.py"
    args = _parser.parse_args(sys.argv[1:] if is_script else [])

    dataset_choice = mo.ui.dropdown(list(make_queries.DATASETS), value=args.dataset, label="dataset")
    mini_choice = mo.ui.checkbox(value=args.mini, label="mini")
    overwrite_choice = mo.ui.checkbox(value=args.overwrite, label="overwrite")
    workers_choice = mo.ui.number(start=1, stop=32, value=args.workers, label="workers")
    overrides_choice = mo.ui.text_area(value="\n".join(args.overrides), label="setting overrides", rows=2)
    run_button = mo.ui.run_button(label="Run")
    mo.vstack([dataset_choice, mini_choice, overwrite_choice, workers_choice, overrides_choice, run_button])
    return (
        args,
        dataset_choice,
        is_script,
        mini_choice,
        overrides_choice,
        overwrite_choice,
        run_button,
        workers_choice,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Run

    In the editor, the step starts only when you press **Run**. The output shows a progress bar
    and these lines:

    - `query images`: how many images were made, how many existed already (kept), and how many
      photos failed. Each failed photo is listed with the reason, for example `MaskMismatchError`
      (the mask file belongs to another photo).
    - `catalog`: the number of query rows and mesh rows written. A mesh gets a row only if at least
      one of its photos became a query.
    - `split`: the size of the train, val, and test parts of each split, and the file.
    - `report`: the report file and the run time.
    """)
    return


@app.cell
def _(
    args,
    dataset_choice,
    is_script,
    load_config,
    make_queries,
    mini_choice,
    mo,
    overrides_choice,
    overwrite_choice,
    run_button,
    workers_choice,
):
    mo.stop(not (is_script or run_button.value), mo.md("Press **Run** to start the step."))
    _overrides = [line.strip() for line in overrides_choice.value.splitlines() if line.strip()]
    report = make_queries.run(
        dataset_choice.value,
        load_config(args.config, _overrides),
        mini=mini_choice.value,
        overwrite=overwrite_choice.value,
        workers=int(workers_choice.value),
    )
    return (report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Summary

    | Field | Meaning |
    |:---|:---|
    | `photos_listed` | Photos in the dataset annotations (before the mini selection) |
    | `queries_prepared` | Photos that became query images and catalog rows |
    | `meshes` | Mesh rows in the catalog |
    | `left_out` | Photos that could not be prepared |
    | `<split>` | train / val / test sizes of each split |
    | `seconds` | Run time of the step |
    """)
    return


@app.cell
def _(mo, report):
    _summary = {
        "photos_listed": report["photos_listed"],
        "queries_prepared": report["queries_prepared"],
        "meshes": report["meshes"],
        "left_out": ", ".join(r["query_id"] for r in report["left_out"]) or "none",
        **{name: " / ".join(str(n) for n in counts.values()) for name, counts in report["splits"].items()},
        "seconds": report["seconds"],
    }
    mo.ui.table([_summary], selection=None)
    return


if __name__ == "__main__":
    app.run()
