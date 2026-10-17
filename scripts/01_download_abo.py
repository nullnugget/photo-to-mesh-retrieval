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
    # 01: Download ABO

    This step downloads the part of the Amazon Berkeley Objects (ABO) dataset that the project
    uses: the metadata, the geometry of the selected 3D meshes, and their catalog photos.
    Pix3D does not need this step.

    | Output | Location |
    |:---|:---|
    | Metadata (18 files, about 95 MB) | `data/raw/abo/metadata/` |
    | Manifest: the selected meshes and photos, one result row per mesh, the report | `data/raw/abo/manifest/` |
    | Mesh geometry, `<model id>.ply`, and its glTF description, `<model id>.gltf.json` | `data/raw/abo/meshes/<class>/` |
    | Catalog photos at full resolution | `data/raw/abo/images/<class>/<image id>.jpg` |

    The class rules in `configs/data.yaml` select the meshes (27 classes, 4,943 meshes, 29,263
    photos). For each mesh, only the byte range with the vertex positions and the triangles is
    downloaded (about 5 % of the `.glb` file); the texture images are not needed.

    A **mini** run takes the first 3 meshes of each class and only their photos, and writes them
    to `test_outputs/download_abo_mini/`. The metadata always goes to `data/raw/abo/metadata/`.

    Files that exist already are skipped, so a stopped run continues where it stopped.

    **How to run** (from the repository root)

    - In the editor: `marimo edit scripts/01_download_abo.py`. Set the options below, then press **Run**.
    - In the terminal: `python scripts/01_download_abo.py --mini` (quick check, about a minute), or
      without `--mini` for everything (about 12 GB; 30 to 60 minutes). `--help` lists all options.

    The logic of the step is in `src/pipeline/download_abo.py`. This file only collects the options
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
    from src.pipeline import download_abo  # noqa: E402

    return add_config_args, argparse, download_abo, load_config, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Options

    | Option | Terminal | Meaning |
    |:---|:---|:---|
    | mini | `--mini` | 3 meshes per class and their photos, into `test_outputs/download_abo_mini/`. |
    | stages | `--stages metadata meshes photos` | The stages to run. The selection of meshes and photos always runs. |
    | workers | `--workers 16` | Number of parallel downloads. Each keeps its connection open. |
    | setting overrides | `--set data.abo.max_models_per_class=50` | Change a value of `configs/*.yaml` for this run only. In the editor: one `KEY=VALUE` per line. |

    From the terminal, the options come from the command line. In the editor, they start at
    their default values.
    """)
    return


@app.cell
def _(add_config_args, argparse, download_abo, mo, sys):
    _parser = argparse.ArgumentParser(
        prog="scripts/01_download_abo.py",
        description="Download the ABO metadata, mesh geometry, and catalog photos of the selected classes.",
    )
    _parser.add_argument("--mini", action="store_true", help="3 meshes per class and their photos, into test_outputs/")
    _parser.add_argument("--stages", nargs="+", choices=download_abo.STAGES, default=list(download_abo.STAGES), help="stages to run")
    _parser.add_argument("--workers", type=int, default=None, help="parallel downloads (default: data.abo.download_workers)")
    add_config_args(_parser)
    is_script = mo.app_meta().mode == "script"  # True when started with "python scripts/01_download_abo.py"
    args = _parser.parse_args(sys.argv[1:] if is_script else [])

    mini_choice = mo.ui.checkbox(value=args.mini, label="mini")
    stages_choice = mo.ui.multiselect(list(download_abo.STAGES), value=args.stages, label="stages")
    workers_choice = mo.ui.number(start=1, stop=64, value=args.workers or 16, label="workers")
    overrides_choice = mo.ui.text_area(value="\n".join(args.overrides), label="setting overrides", rows=2)
    run_button = mo.ui.run_button(label="Run")
    mo.vstack([mini_choice, stages_choice, workers_choice, overrides_choice, run_button])
    return args, is_script, mini_choice, overrides_choice, run_button, stages_choice, workers_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Run

    In the editor, the step starts only when you press **Run**. The output shows a progress bar
    for each stage and these lines:

    - `selection`: the number of selected meshes and photos, and of classes.
    - `meshes`: how many meshes were downloaded (`done`), existed already (`skipped`), or failed,
      and the downloaded megabytes.
    - `photos`: the same for the photos.
    - `report`: the report file and the run time.
    """)
    return


@app.cell
def _(
    args,
    download_abo,
    is_script,
    load_config,
    mini_choice,
    mo,
    overrides_choice,
    run_button,
    stages_choice,
    workers_choice,
):
    mo.stop(not (is_script or run_button.value), mo.md("Press **Run** to start the step."))
    _overrides = [line.strip() for line in overrides_choice.value.splitlines() if line.strip()]
    # In script mode the terminal value of --workers wins (None = the configured default).
    _workers = args.workers if is_script else int(workers_choice.value)
    report = download_abo.run(
        load_config(args.config, _overrides),
        mini=mini_choice.value,
        stages=tuple(stages_choice.value),
        workers=_workers,
    )
    return (report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Summary

    One row per class:

    | Field | Meaning |
    |:---|:---|
    | `class` | Our class name |
    | `meshes` | Selected meshes of the class |
    | `photos` | Selected photos of the class (a photo that several meshes show counts once, in the class of the first mesh) |

    Below the table: the result counts of each stage, and the failed files with the reason.
    """)
    return


@app.cell
def _(mo, report):
    _rows = [{"class": c, **counts} for c, counts in report["selection"]["classes"].items()]
    _lines = []
    for _stage in ("meshes", "photos"):
        if _stage in report:
            _lines.append(f"**{_stage}:** {report[_stage]['status']}, {report[_stage]['bytes_downloaded'] / 1e6:.1f} MB, {report[_stage]['seconds']} s")
    if report.get("meshes", {}).get("failed"):
        _lines.append("**failed meshes:** " + "; ".join(f"`{f['mesh_id']}` ({f['message']})" for f in report["meshes"]["failed"]))
    if report.get("photos", {}).get("failed"):
        _lines.append(f"**failed photos:** {len(report['photos']['failed'])} ({report['photos']['failure_kinds']})")
    mo.vstack([mo.ui.table(_rows, selection=None), mo.md("\n\n".join(_lines) or "No download stage ran.")])
    return


if __name__ == "__main__":
    app.run()
