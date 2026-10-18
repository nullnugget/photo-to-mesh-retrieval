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
    # 03: ABO Check

    This notebook checks the ABO files that the download step downloaded (`scripts/01_download_abo.py`):

    0. optionally, a mini run of the download step (3 meshes per class and their photos);
    1. the selection: meshes and photos per class, and the results of the download;
    2. the mesh geometry: every PLY file against the ABO metadata (vertex count, face count, size);
    3. the orientation: silhouettes of one mesh per class from the front, the side, and the top,
       next to the main catalog photo of the product;
    4. the photos: a contact sheet per class, and their sizes;
    5. the masks of the segment step (`scripts/02_segment_photos.py --dataset abo`): which photos the
       cleaning rules keep or remove, and why, with the masks drawn on the photos (section 7).

    **How to run**

    - Full download: run `python scripts/01_download_abo.py` first. Mini download: press the button
      in section 2 (in the editor), or run the script with `--mini`.
    - Then, from the repository root: `marimo edit notebooks/03_abo_check.py` (interactively), or
      `python notebooks/03_abo_check.py` (as a script; it checks the full download if it exists,
      otherwise the mini download).
    - Section 7 needs the segment step: a sample to review the cleaning rules with
      `python scripts/02_segment_photos.py --dataset abo --mini --mini-count 20` (20 random photos per
      class, about 5 minutes on the GPU), or the full run with `--dataset abo`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules, including the download step and
    the segment step.
    """)
    return


@app.cell
def _():
    import csv
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
    from pytorch3d.io import load_ply  # noqa: E402

    from src import paths  # noqa: E402
    from src.config import load_config  # noqa: E402
    from src.data import abo  # noqa: E402
    from src.data.catalog import read_table  # noqa: E402
    from src.pipeline import download_abo, segment_photos  # noqa: E402
    from src.preprocess import segment  # noqa: E402

    SAMPLE_SEED = 0  # fixed seed for the random samples below
    REVIEW_TILES = 18  # photos per review grid in section 7
    return (
        Counter,
        Image,
        ImageDraw,
        PROJECT_ROOT,
        REVIEW_TILES,
        SAMPLE_SEED,
        abo,
        csv,
        download_abo,
        json,
        load_config,
        load_ply,
        np,
        paths,
        random,
        read_table,
        segment,
        segment_photos,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Mini run (optional) and the download to check

    The button runs the download step for 3 meshes per class and their photos
    (`download_abo.run(..., mini=True)`, the same function as `scripts/01_download_abo.py --mini`).
    It writes to `test_outputs/download_abo_mini/` (the metadata to `data/raw/abo/metadata/`) and
    takes about a minute the first time; files that exist are skipped. Then choose which download
    the rest of the notebook checks: `full` (`data/raw/abo/`) or `mini`. When the notebook runs as a
    script, the button is not pressed.
    """)
    return


@app.cell
def _(mo):
    mini_button = mo.ui.run_button(label="Make the mini download")
    mini_button
    return (mini_button,)


@app.cell
def _(download_abo, load_config, mini_button):
    # Runs only when the button is pressed; otherwise mini_report is None and nothing is downloaded.
    mini_report = download_abo.run(load_config(), mini=True) if mini_button.value else None
    return (mini_report,)


@app.cell
def _(download_abo, mini_report, mo):
    _ = mini_report  # makes this cell run again after a mini run, so that a new download appears in the list
    DOWNLOADS = {
        name: dirs
        for name, dirs in {"full": download_abo.output_dirs(False), "mini": download_abo.output_dirs(True)}.items()
        if (dirs["manifest"] / "download_report.json").exists()
    }
    download_choice = mo.ui.dropdown(list(DOWNLOADS), value=next(iter(DOWNLOADS)), label="download to check")
    download_choice
    return DOWNLOADS, download_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Selection and download results

    The next cell reads the manifest (the selected meshes and photos), the result row of each
    mesh, and the report of the download step.

    | Field | Meaning |
    |:---|:---|
    | `class` | Our class name |
    | `meshes` | Selected meshes of the class |
    | `photos` | Selected photos of the class (a photo that several meshes show counts once) |
    | `meshes_ok` | Meshes whose PLY file was written (in this run or an earlier one) |

    Below the table: the result counts of the stages, the share of the full `.glb` bytes that the
    geometry download needed, and the photos that several meshes show.
    """)
    return


@app.cell
def _(Counter, DOWNLOADS, abo, csv, download_choice, json, mo):
    dirs = DOWNLOADS[download_choice.value]
    meshes, photos = abo.read_manifest(dirs["manifest"])
    report = json.loads((dirs["manifest"] / "download_report.json").read_text())
    with open(dirs["manifest"] / "mesh_files.csv", newline="") as _f:
        mesh_results = {row["mesh_id"]: row for row in csv.DictReader(_f)}

    _ok = Counter(m.category for m in meshes if mesh_results.get(m.mesh_id, {}).get("status") in ("done", "skipped"))
    _rows = [{"class": c, **counts, "meshes_ok": _ok[c]} for c, counts in abo.class_counts(meshes, photos).items()]
    _shared = [p for p in photos if len(p.mesh_ids) > 1]
    _cross = [p for p in _shared if len({m.split("/")[1] for m in p.mesh_ids}) > 1]
    print("SELECTION", {"meshes": len(meshes), "photos": len(photos), "shared_photos": len(_shared), "cross_class": len(_cross)})
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                f"**meshes:** {report.get('meshes', {}).get('status')}, share of the full files: "
                f"{report.get('meshes', {}).get('share_of_full_files')}  \n"
                f"**photos:** {report.get('photos', {}).get('status')}  \n"
                f"**photos shown by several meshes:** {len(_shared)} ({len(_cross)} of them by meshes of different classes)"
            ),
        ]
    )
    return dirs, mesh_results, meshes, photos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Mesh geometry against the metadata

    The next cell reads every PLY file with PyTorch3D (the reader that the renderer uses) and
    compares it with the ABO metadata (`3dmodels.csv`):

    | Measure | Meaning | Expected |
    |:---|:---|:---|
    | `checked` | PLY files read | all meshes with a file |
    | `missing` | Selected meshes without a PLY file | 0 |
    | `vertex_count_differs` | Files whose vertex count differs from the metadata | 0 |
    | `face_count_differs` | Files whose face count differs from the metadata | 0 |
    | `size_differs` | Files whose bounding box differs by more than 1 % from the metadata size (extent_x, y, z). The size is measured after the node placement, so a wrong placement shows here | 0 |
    | `not_finite` | Files with a NaN or infinite coordinate | 0 |

    The table below the measures lists the meshes that differ (empty when all agree). The full
    download has about 5,000 files, so this cell then takes about a minute.
    """)
    return


@app.cell
def _(dirs, download_abo, load_ply, meshes, mo, np):
    _checks = {"checked": 0, "missing": 0, "vertex_count_differs": 0, "face_count_differs": 0, "size_differs": 0, "not_finite": 0}
    _differences = []
    for _mesh in meshes:
        _path = download_abo.mesh_file(dirs["meshes"], _mesh)
        if not _path.exists():
            _checks["missing"] += 1
            continue
        _vertices, _faces = load_ply(_path)
        _vertices = _vertices.numpy()
        _checks["checked"] += 1
        _size_ratio = (_vertices.max(axis=0) - _vertices.min(axis=0)) / np.maximum(np.array(_mesh.extent), 1e-9)
        _problems = {
            "vertex_count_differs": len(_vertices) != _mesh.num_vertices,
            "face_count_differs": len(_faces) != _mesh.num_faces,
            "size_differs": not np.allclose(_size_ratio, 1.0, atol=0.01),
            "not_finite": not np.isfinite(_vertices).all(),
        }
        for _name, _bad in _problems.items():
            _checks[_name] += int(_bad)
        if any(_problems.values()):
            _differences.append(
                {"mesh_id": _mesh.mesh_id, "vertices": len(_vertices), "metadata_vertices": _mesh.num_vertices,
                 "faces": len(_faces), "metadata_faces": _mesh.num_faces, "size_ratio_xyz": str(np.round(_size_ratio, 3))}
            )
    print("GEOMETRY", _checks)
    mo.vstack([mo.ui.table([_checks], selection=None), mo.ui.table(_differences, selection=None) if _differences else mo.md("No differences.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Orientation

    The renderer expects every mesh with +Y up and the front towards +Z (the Pix3D convention,
    which ABO also uses). The next cell draws, for one random mesh per class, three silhouettes
    in grey: from the front (looking along −Z), from the right side (looking along −X), and from
    the top (looking down along −Y, with the front at the bottom). The main catalog photo of
    the product is on the left. The three silhouettes of one mesh have the same scale: the
    largest side of the mesh fills the tile.

    What to look for: in the front silhouette, the object stands upright and shows the same face
    as the catalog photo (most product photos show the front, sometimes at an angle). The
    silhouettes are flat projections of all triangles; they show the outline, not the shading.
    """)
    return


@app.cell
def _(Image, ImageDraw, SAMPLE_SEED, dirs, download_abo, load_ply, meshes, np, photos, random):
    def silhouette(vertices, faces, axes, flip_horizontal, scale, size=160, max_faces=40000):
        """Project the triangles onto two axes and fill them in grey on white, centred in the tile.

        Args:
            vertices, faces: The mesh (numpy arrays).
            axes: (horizontal axis, vertical axis) indices into x, y, z. The vertical axis points up.
            flip_horizontal: True when the horizontal image axis runs against the world axis.
            scale: Pixels per mesh unit. The three views of one mesh use the same scale, so that
                their sizes can be compared.
            max_faces: Larger meshes are drawn with a random subset of faces. The outline stays
                correct, but large flat areas can show small white specks.
        """
        if len(faces) > max_faces:
            faces = faces[np.random.default_rng(0).choice(len(faces), max_faces, replace=False)]
        points = vertices[:, list(axes)].astype(np.float64)
        points[:, 0] *= -1.0 if flip_horizontal else 1.0
        points[:, 1] *= -1.0  # image rows grow downwards
        centre = (points.min(axis=0) + points.max(axis=0)) / 2
        pixels = (points - centre) * scale + size / 2
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)
        for triangle in pixels[faces]:
            draw.polygon([tuple(p) for p in triangle], fill=(120, 120, 120))
        return image

    def orientation_row(mesh, photo_path, tile=160):
        """One row: catalog photo, front, side, and top silhouettes of one mesh."""
        vertices, faces = (t.numpy() for t in load_ply(download_abo.mesh_file(dirs["meshes"], mesh)))
        row = Image.new("RGB", (4 * tile, tile), "white")
        with Image.open(photo_path) as photo:
            photo = photo.convert("RGB")
            photo.thumbnail((tile, tile))
            row.paste(photo, ((tile - photo.width) // 2, (tile - photo.height) // 2))
        scale = (tile - 10) / max(vertices.max(axis=0) - vertices.min(axis=0))  # the largest side fills the tile
        row.paste(silhouette(vertices, faces, (0, 1), False, scale, tile), (tile, 0))  # front: x right, y up
        row.paste(silhouette(vertices, faces, (2, 1), True, scale, tile), (2 * tile, 0))  # right side: -z right, y up
        row.paste(silhouette(vertices, faces, (0, 2), False, scale, tile), (3 * tile, 0))  # top: x right, +z (front) down
        return row

    _rng = random.Random(SAMPLE_SEED)
    _photo_by_id = {p.image_id: p for p in photos}
    _rows = []
    for _category in sorted({m.category for m in meshes}):
        _pool = [m for m in meshes if m.category == _category and download_abo.mesh_file(dirs["meshes"], m).exists()]
        if not _pool:
            continue
        _mesh = _rng.choice(_pool)
        _photo = _photo_by_id[_mesh.image_ids[0]]  # the main photo comes first in the listing
        _rows.append((_mesh.mesh_id, orientation_row(_mesh, download_abo.photo_file(dirs["photos"], _photo))))
    orientation_sheet = Image.new("RGB", (4 * 160, 160 * len(_rows)), "white")
    for _i, (_, _row) in enumerate(_rows):
        orientation_sheet.paste(_row, (0, 160 * _i))
    print("ORIENTATION", {"rows": len(_rows), "meshes": [mesh_id for mesh_id, _ in _rows]})
    orientation_sheet
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Photos

    The next cell shows 8 random photos per class (one class per row), and measures all photos:

    | Measure | Meaning |
    |:---|:---|
    | `files` | Selected photos with a file on disk |
    | `missing` | Selected photos without a file |
    | `unreadable` | Files that Pillow cannot open |
    | `size_differs` | Files whose pixel size differs from the metadata |
    | `megabytes` | Total size of the files |
    | `median_kilobytes` | Median file size |

    What to look for in the contact sheet: the three kinds of catalog photos (white-background
    product photos, room photos, close-ups of details). The segmentation step will remove the
    close-ups later.
    """)
    return


@app.cell
def _(Image, ImageDraw, SAMPLE_SEED, dirs, download_abo, mo, np, photos, random):
    _measures = {"files": 0, "missing": 0, "unreadable": 0, "size_differs": 0}
    _sizes = []
    for _photo in photos:
        _path = download_abo.photo_file(dirs["photos"], _photo)
        if not _path.exists():
            _measures["missing"] += 1
            continue
        _measures["files"] += 1
        _sizes.append(_path.stat().st_size)
        try:
            with Image.open(_path) as _image:
                _measures["size_differs"] += int(_image.size != (_photo.width, _photo.height))
        except OSError:
            _measures["unreadable"] += 1
    _measures["megabytes"] = round(sum(_sizes) / 1e6, 1)
    _measures["median_kilobytes"] = round(float(np.median(_sizes)) / 1e3) if _sizes else 0
    print("PHOTOS", _measures)

    _rng = random.Random(SAMPLE_SEED)
    _categories = sorted({p.category for p in photos})
    _tile, _label = 112, 110
    _sheet = Image.new("RGB", (_label + 8 * _tile, len(_categories) * _tile), "white")
    _draw = ImageDraw.Draw(_sheet)
    for _r, _category in enumerate(_categories):
        _pool = [p for p in photos if p.category == _category and download_abo.photo_file(dirs["photos"], p).exists()]
        _draw.text((4, _r * _tile + _tile // 2 - 6), _category, fill="black")
        for _c, _photo in enumerate(_rng.sample(_pool, min(8, len(_pool)))):
            with Image.open(download_abo.photo_file(dirs["photos"], _photo)) as _image:
                _image = _image.convert("RGB")
                _image.thumbnail((_tile, _tile))
                _sheet.paste(_image, (_label + _c * _tile + (_tile - _image.width) // 2, _r * _tile + (_tile - _image.height) // 2))
    mo.vstack([mo.ui.table([_measures], selection=None), _sheet])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Masks and cleaning (segment step)

    The segment step finds each ABO photo's object with Grounding DINO (prompt: the class name, for
    example "sofa."), makes the SAM 2.1 mask from the most confident box, and applies the cleaning rules
    (`configs/segment.yaml`, `segment.cleaning`). A photo is **removed** for one of these reasons:

    | Reason | Rule | Typical photo |
    |:---|:---|:---|
    | `no_detection` | No box reaches the detector threshold (`box_threshold`, 0.35) | colour swatch, label, text-only image |
    | `close_up` | The box touches more than `max_edges_touched` (2) photo edges | detail of a material, a drawer, a leg |
    | `mask_too_small` | The mask covers less than `min_mask_area` (5 %) of the photo | the object is tiny, or the detector chose a small part |
    | `mask_too_large` | The mask covers more than `max_mask_area` (90 %) of the photo | swatch or texture that fills the frame |

    Choose the outputs to check: `sample` (`--mini --mini-count 20`, in `test_outputs/segment_mini/`) or
    `full` (`data/derived/masks/`).
    """)
    return


@app.cell
def _(mo, segment_photos):
    SEGMENT_OUTPUTS = {
        name: root
        for name, root in {"full": segment_photos.output_root(False), "sample": segment_photos.output_root(True)}.items()
        if segment_photos.results_file("abo", root).exists()
    }
    mo.stop(not SEGMENT_OUTPUTS, mo.md("No ABO masks yet. Run `python scripts/02_segment_photos.py --dataset abo --mini --mini-count 20`."))
    segment_choice = mo.ui.dropdown(list(SEGMENT_OUTPUTS), value=next(iter(SEGMENT_OUTPUTS)), label="masks to check")
    segment_choice
    return SEGMENT_OUTPUTS, segment_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.1 Kept and removed photos per class

    | Field | Meaning |
    |:---|:---|
    | `class` | Our class name (the last row adds up all classes) |
    | `photos` | Photos segmented |
    | `kept`, `kept_share` | Photos that pass every rule, and their share |
    | `no_detection`, `close_up`, `mask_too_small`, `mask_too_large` | Removed photos per reason (see the table above) |
    | `median_detector_score` | Median confidence of the chosen box (photos with a detection) |
    | `median_mask_area` | Median share of the photo covered by the mask (kept photos) |

    A class with a low `kept_share` or many `no_detection` rows may need another prompt (`segment.prompts`).
    """)
    return


@app.cell
def _(Counter, SEGMENT_OUTPUTS, abo, download_abo, mo, np, paths, read_table, segment_choice, segment_photos):
    SEGMENT_ROOT = SEGMENT_OUTPUTS[segment_choice.value]
    _, _photos = abo.read_manifest(paths.ABO_MANIFEST_DIR)
    _path_of = {p.image_id: download_abo.photo_file(paths.ABO_IMAGES_DIR, p) for p in _photos}
    # One record per segmented photo: the results row, the photo file, and the mask file.
    SEGMENTED = [
        {**r, "image_id": r["photo_id"].split("/")[-1], "photo_file": _path_of[r["photo_id"].split("/")[-1]],
         "mask_file": SEGMENT_ROOT / "abo" / r["category"] / (r["photo_id"].split("/")[-1] + ".png")}
        for r in read_table(segment_photos.results_file("abo", SEGMENT_ROOT))
        if r["status"] == "done"
    ]
    REASONS = ["no_detection", "close_up", "mask_too_small", "mask_too_large"]

    def _summary(name, rows):
        reasons = Counter(r["reason"] for r in rows if r["verdict"] == "removed")
        scores = [float(r["detector_score"]) for r in rows if r["detector_score"] not in ("", None)]
        areas = [float(r["mask_area"]) for r in rows if r["verdict"] == "kept"]
        kept = sum(r["verdict"] == "kept" for r in rows)
        return {"class": name, "photos": len(rows), "kept": kept, "kept_share": round(kept / max(len(rows), 1), 2),
                **{reason: reasons[reason] for reason in REASONS},
                "median_detector_score": round(float(np.median(scores)), 3) if scores else None,
                "median_mask_area": round(float(np.median(areas)), 3) if areas else None}

    _table = [_summary(c, [r for r in SEGMENTED if r["category"] == c]) for c in sorted({r["category"] for r in SEGMENTED})]
    _table.append(_summary("all", SEGMENTED))
    print("CLEANING", _table[-1])
    mo.ui.table(_table, selection=None, page_size=30)
    return REASONS, SEGMENTED


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7.2 Review grid

    Each tile shows a photo with its mask in **red** and the detector box in **yellow**. The caption gives
    the class, the photo size, the verdict and reason, the detector confidence, and the mask area. Choose
    a class, a verdict, and a reason; the grid shows up to 18 random photos of the selection (fixed seed).

    What to look for: kept photos should show the whole object (or most of it) with a mask on the object;
    removed photos should be swatches, details, labels, or photos without the object. A good photo that
    was removed, or a close-up that was kept, shows which rule to change.
    """)
    return


@app.cell
def _(REASONS, SEGMENTED, mo):
    review_class = mo.ui.dropdown(["all"] + sorted({r["category"] for r in SEGMENTED}), value="all", label="class")
    review_verdict = mo.ui.dropdown(["all", "kept", "removed"], value="kept", label="verdict")
    review_reason = mo.ui.dropdown(["all"] + REASONS, value="all", label="reason (removed photos)")
    mo.hstack([review_class, review_verdict, review_reason], justify="start")
    return review_class, review_reason, review_verdict


@app.cell
def _(Image, ImageDraw, REVIEW_TILES, SAMPLE_SEED, SEGMENTED, mo, np, random, review_class, review_reason, review_verdict, segment):
    _rows = [
        r for r in SEGMENTED
        if review_class.value in ("all", r["category"])
        and review_verdict.value in ("all", r["verdict"])
        and review_reason.value in ("all", r["reason"])
    ]
    _rows = random.Random(SAMPLE_SEED).sample(_rows, min(REVIEW_TILES, len(_rows)))

    def review_tile(row, size=220):
        """The photo with the mask (red) and the box (yellow), scaled to fit `size` pixels, and its caption."""
        with Image.open(row["photo_file"]) as photo:
            pixels = np.asarray(photo.convert("RGB")).astype(float)
        if row["mask_file"].exists():
            mask = segment.read_mask(row["mask_file"])
            pixels[mask] = pixels[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
        tile = Image.fromarray(pixels.astype(np.uint8))
        if row["box"]:
            ImageDraw.Draw(tile).rectangle([float(v) for v in row["box"].split()], outline=(255, 200, 0), width=max(2, tile.width // 120))
        caption = (f"{row['category']} {tile.width}x{tile.height}  \n{row['verdict']} {row['reason']}  \n"
                   f"score {row['detector_score'] or '-'}, area {row['mask_area'] or '-'}")
        tile.thumbnail((size, size))
        return mo.vstack([mo.image(tile), mo.md(caption)])

    mo.hstack([review_tile(r) for r in _rows], wrap=True, justify="start") if _rows else mo.md("No photos in this selection.")
    return


if __name__ == "__main__":
    app.run()
