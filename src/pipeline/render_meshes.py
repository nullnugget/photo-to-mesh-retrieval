"""The render step: 25 grey views of every gallery mesh of one dataset.

For every mesh row of the dataset in the catalog (`data/derived/catalog/meshes.csv`), this step:

1. loads the mesh file, turns its front to +Z (`render.front_axis`), and scales it to the unit sphere;
2. renders it from the 25 cameras of the rig with the active light setup (`render.lighting`);
3. saves the cropped views (`render.output_size`, 224 x 224) as `view_00.png` ... `view_24.png` in the
   mesh's `render_dir` (for example `data/derived/renders/pix3d/chair/IKEA_POANG_2/`).

Views of another output size than the main image size (`data.image_size`) go to a folder of their own,
with the same subfolders, for example `--set render.output_size=448` into `data/derived/renders_448/`
(`paths.size_suffix`). The renders of the main size are not touched.

Safe to stop and start again:

- The views of a mesh are written into `<render_dir>.part/` and the folder is renamed at the end,
  so a folder without `.part` always holds all 25 views. Existing folders are skipped.
- The settings that made the renders are saved in `render_config.yaml` next to them. A new run with
  different settings stops with an error instead of mixing old and new renders; `overwrite=True`
  renders everything again.

Outputs (normal run):
    data/derived/renders/<dataset>/<category>/<name>/view_XX.png    the views
    data/derived/renders/render_config.yaml                          the render settings
    data/derived/renders/<dataset>_render_report.json                totals of all runs, failures
    data/derived/renders/<dataset>_render_results.csv                one row per mesh (see RESULT_COLUMNS)

A mini run renders 2 meshes per category into `test_outputs/render_mini/`, so that the real
outputs are not touched.

The ABO meshes get catalog rows from the ABO query step; until then, this step has no ABO rows.
"""

import csv
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src import paths
from src.config import load_yaml, save_config
from src.data.catalog import read_table
from src.render.renderer import MeshRenderer

MINI_MESHES_PER_CATEGORY = 2
VIEW_NAME = "view_{:02d}.png"
# One row per mesh in <dataset>_render_results.csv. A mesh that a later run skips keeps the row of
# the run that rendered it, so the totals in the report always cover all renders on disk.
RESULT_COLUMNS = ["mesh_id", "status", "views_touching_border", "overflow_retries", "seconds", "message"]


class RenderSettingsChanged(RuntimeError):
    """The existing renders were made with other settings (see `render_config.yaml`)."""


def output_root(mini: bool, suffix: str = "") -> Path:
    """Return the folder that holds the renders: the real one, or a test folder for a mini run.

    `suffix` (`paths.size_suffix`) selects the folder of another output size, for example `renders_448`.
    """
    return paths.sized(paths.TEST_OUTPUTS_DIR / "render_mini" / "renders" if mini else paths.RENDERS_DIR, suffix)


def render_folder(mesh_row: dict, root: Path) -> Path:
    """Return the folder of one mesh's views. For the real run this is the catalog's `render_dir`."""
    relative = Path(mesh_row["render_dir"]).relative_to(paths.RENDERS_DIR.relative_to(paths.ROOT))
    return root / relative  # for example <root>/pix3d/chair/IKEA_POANG_2


def render_report_file(dataset: str, root: Path = paths.RENDERS_DIR) -> Path:
    """Return the report file of the render step for one dataset."""
    return root / f"{dataset}_render_report.json"


def render_results_file(dataset: str, root: Path = paths.RENDERS_DIR) -> Path:
    """Return the per-mesh result table of the render step for one dataset."""
    return root / f"{dataset}_render_results.csv"


def _merge_results(path: Path, new_rows: list[dict]) -> list[dict]:
    """Update the result table with the rows of this run; return all rows of the table.

    - A mesh of this run gets its new row, but a skipped mesh keeps its earlier row (the skipped
      row has no values).
    - A mesh that is not in this run keeps its earlier row (for example, after a run on some
      categories only).
    """
    merged: dict[str, dict] = {}
    if path.exists():
        with open(path, newline="") as f:
            merged = {row["mesh_id"]: row for row in csv.DictReader(f)}
    for row in new_rows:
        if row["status"] != "skipped" or row["mesh_id"] not in merged:
            merged[row["mesh_id"]] = row
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(merged.values())
    return list(merged.values())


def _check_settings(render_cfg: dict, root: Path, overwrite: bool) -> None:
    """Stop if earlier renders in `root` were made with other settings; save the current settings."""
    record = root / paths.RENDER_CONFIG_FILE.name
    if record.exists() and not overwrite and load_yaml(record) != {"render": render_cfg}:
        raise RenderSettingsChanged(
            f"{record} differs from the current render settings. Render again with overwrite "
            "(all views are replaced), or restore the settings."
        )
    root.mkdir(parents=True, exist_ok=True)
    save_config({"render": render_cfg}, record)


def _save_views(images, folder: Path) -> None:
    """Write the views into `<folder>.part/`, then rename it to `folder` (complete or absent)."""
    partial = folder.with_name(folder.name + ".part")
    if partial.exists():
        shutil.rmtree(partial)  # left over from an interrupted run
    partial.mkdir(parents=True)
    for index, image in enumerate(images):
        Image.fromarray(image).save(partial / VIEW_NAME.format(index))
    if folder.exists():
        shutil.rmtree(folder)  # only with overwrite: the old views are replaced
    partial.rename(folder)


def run(dataset: str, cfg: dict, mini: bool = False, overwrite: bool = False, device: str = "cuda") -> dict:
    """Render all gallery meshes of one dataset, and return the report (also saved as JSON).

    Args:
        dataset: "pix3d" (or "abo", once the ABO meshes are in the catalog).
        cfg: The merged settings; the step reads `cfg["render"]` (and `data.image_size`, the main size).
        mini: True = 2 meshes per category, into test_outputs/render_mini/.
        overwrite: True = render every mesh again, also with changed settings.
        device: "cuda" or "cpu".

    Raises:
        ValueError: If the catalog has no mesh rows of the dataset.
        RenderSettingsChanged: If earlier renders used other settings and `overwrite` is False.
    """
    render_cfg = cfg["render"]
    root = output_root(mini, paths.size_suffix(render_cfg["output_size"], cfg["data"]["image_size"]))
    meshes = [m for m in read_table(paths.MESHES_CSV) if m["dataset"] == dataset]
    if not meshes:
        raise ValueError(f"the catalog has no mesh rows of dataset {dataset!r}")
    if mini:
        taken: Counter = Counter()
        meshes = [m for m in meshes if (taken.update([m["category"]]) or taken[m["category"]] <= MINI_MESHES_PER_CATEGORY)]
    _check_settings(render_cfg, root, overwrite)

    renderer = MeshRenderer.from_config(render_cfg, device=device)
    front_axis = render_cfg["front_axis"][dataset]
    print(f"{dataset}: {len(meshes)} meshes{' (mini)' if mini else ''}; light {renderer.light_setup.name}; "
          f"{len(renderer.poses)} views each -> {root.relative_to(paths.ROOT)}")

    rows = []
    start = time.perf_counter()
    for mesh in tqdm(meshes, desc="render", unit="mesh"):
        row = {"mesh_id": mesh["mesh_id"], "status": "skipped", "views_touching_border": 0, "overflow_retries": 0, "seconds": 0.0, "message": ""}
        rows.append(row)
        folder = render_folder(mesh, root)
        if folder.exists() and not overwrite:
            continue
        retries_before, mesh_start = renderer.overflow_retries, time.perf_counter()
        try:
            views = renderer.render_file(mesh["mesh_path"], front_axis=front_axis)
        except (ValueError, RuntimeError, OSError) as error:  # an unreadable or empty mesh, or out of GPU memory
            row.update(status="failed", message=f"{type(error).__name__}: {error}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # free the memory of the failed mesh before the next one
            continue
        _save_views(views.images, folder)
        row.update(status="done", views_touching_border=int(views.touches_border.sum()),
                   overflow_retries=renderer.overflow_retries - retries_before, seconds=round(time.perf_counter() - mesh_start, 2))

    this_run = Counter(row["status"] for row in rows)
    all_rows = _merge_results(render_results_file(dataset, root), rows)  # totals over every run
    report = {
        "dataset": dataset,
        "mini": mini,
        "light_setup": renderer.light_setup.name,
        "views_per_mesh": len(renderer.poses),
        "meshes": len(meshes),
        "this_run": dict(this_run),
        "rendered": sum(row["status"] == "done" for row in all_rows),  # meshes with views, from any run
        "failed": [{"mesh_id": row["mesh_id"], "message": row["message"]} for row in all_rows if row["status"] == "failed"],
        "views_touching_border": sum(int(row["views_touching_border"]) for row in all_rows),  # expected 0
        "overflow_retries": sum(int(row["overflow_retries"]) for row in all_rows),  # batches rasterized again
        "seconds_this_run": round(time.perf_counter() - start, 1),
    }
    report_path = render_report_file(dataset, root)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"render: this run {dict(this_run)}; {report['rendered']} of {len(meshes)} meshes rendered; "
          f"views touching the border: {report['views_touching_border']}; overflow retries: {report['overflow_retries']}; "
          f"{report['seconds_this_run']} s -> {report_path.relative_to(paths.ROOT)}")
    return report
