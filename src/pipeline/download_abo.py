"""The ABO download step: metadata, the geometry of the selected meshes, and their catalog photos.

Stages (each can run alone; each skips the files that exist already, so a stopped run continues):

1. metadata: the 18 ABO metadata files (about 95 MB) into `data/raw/abo/metadata/`.
2. selection: apply the class rules of `configs/data.yaml`, and write the manifest (the selected
   meshes and photos) into `data/raw/abo/manifest/`. This stage always runs; it takes seconds.
3. meshes: for each selected mesh, download only its geometry (see `src/data/glb.py`):
   - 2 small range requests: the GLB header and the JSON chunk;
   - 1 range request: the byte range with the vertex positions and triangle indices;
   - save `meshes/<class>/<model id>.ply` (the placed geometry) and `<model id>.gltf.json` (the
     JSON chunk, to trace the PLY file back to its source).
4. photos: download each selected photo at full resolution into `images/<class>/<image id>.jpg`.
   A photo that several meshes show is downloaded once, into the folder of the first mesh.

Every mesh gets one row in `manifest/mesh_files.csv` with the result: status, the decoded vertex
and face counts, and the downloaded bytes. The check notebook compares the counts with the ABO
metadata. The run report goes to `manifest/download_report.json`.

A mini run takes the first 3 meshes of each class (and only their photos), and writes the
manifest, meshes, and photos to `test_outputs/download_abo_mini/`. The metadata files always go to
`data/raw/abo/metadata/`: they are the same files in both cases and are needed for the real run.
"""

import csv
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src import paths
from src.data import abo, glb
from src.data.http import Fetcher

STAGES = ("metadata", "meshes", "photos")
MINI_MESHES_PER_CLASS = 3
DONE, SKIPPED, FAILED = "done", "skipped", "failed"
MESH_FILE_COLUMNS = ["mesh_id", "status", "num_vertices", "num_faces", "bytes", "file_bytes", "message"]


def output_dirs(mini: bool) -> dict[str, Path]:
    """Return the output folders: the real ones, or a separate test folder for a mini run."""
    base = paths.TEST_OUTPUTS_DIR / "download_abo_mini" if mini else paths.ABO_DIR
    return {
        "metadata": paths.ABO_METADATA_DIR,  # shared by both modes (see the module description)
        "manifest": base / "manifest",
        "meshes": base / "meshes",
        "photos": base / "images",
    }


def download_metadata(fetcher: Fetcher, folder: Path) -> dict:
    """Download the metadata files that are not in `folder` yet. Returns {"downloaded", "present", "bytes"}."""
    downloaded, present, total = 0, 0, 0
    for name, bucket_path in tqdm(abo.METADATA_FILES.items(), desc="metadata", unit="file"):
        target = folder / name
        if target.exists():
            present += 1
            continue
        total += fetcher.save(f"{abo.BUCKET_URL}/{bucket_path}", target)
        downloaded += 1
    return {"downloaded": downloaded, "present": present, "bytes": total}


def select_from_metadata(folder: Path, data_cfg: dict, mini: bool) -> tuple[list, list]:
    """Read the metadata, apply the class rules, and (for a mini run) keep 3 meshes per class."""
    abo_cfg = data_cfg["abo"]
    meshes, photos = abo.select(
        abo.read_models(folder / "3dmodels.csv.gz"),
        abo.read_model_listings(sorted(folder.glob("listings_*.json.gz"))),
        abo.read_images(folder / "images.csv.gz"),
        abo_cfg["class_rules"],
        abo_cfg["max_models_per_class"],
    )
    if mini:
        taken: Counter = Counter()
        meshes = [m for m in meshes if (taken.update([m.category]) or taken[m.category] <= MINI_MESHES_PER_CLASS)]
        kept_ids = {m.mesh_id for m in meshes}
        photos = [p for p in photos if any(mesh_id in kept_ids for mesh_id in p.mesh_ids)]
        for p in photos:
            p.mesh_ids = [mesh_id for mesh_id in p.mesh_ids if mesh_id in kept_ids]
            p.category = next(m.category for m in meshes if m.mesh_id == p.mesh_ids[0])
    return meshes, photos


def mesh_file(folder: Path, mesh: abo.AboMesh) -> Path:
    """Return the PLY path of one mesh: `<folder>/<class>/<model id>.ply`."""
    return folder / mesh.category / f"{mesh.model_id}.ply"


def download_mesh(fetcher: Fetcher, mesh: abo.AboMesh, folder: Path) -> dict:
    """Download the geometry of one mesh and save it as PLY, plus its glTF JSON. Returns one result row."""
    ply_path = mesh_file(folder, mesh)
    json_path = ply_path.with_suffix(".gltf.json")
    row = {"mesh_id": mesh.mesh_id, "status": SKIPPED, "num_vertices": "", "num_faces": "", "bytes": 0, "file_bytes": "", "message": ""}
    if ply_path.exists() and json_path.exists():
        return row
    url = abo.model_url(mesh.glb_path)
    try:
        # 1. Header and JSON chunk.
        total_length, json_length = glb.parse_header(fetcher.get(url, 0, glb.HEADER_BYTES))
        json_bytes = fetcher.get(url, glb.HEADER_BYTES, glb.HEADER_BYTES + json_length)
        gltf = json.loads(json_bytes)
        # 2. The geometry range of the BIN chunk, then the decoded and placed geometry.
        start, end = glb.geometry_byte_range(gltf)
        bin_start = glb.bin_chunk_start(json_length)
        data = fetcher.get(url, bin_start + start, bin_start + end)
        vertices, faces = glb.decode_geometry(gltf, data, data_start=start)
        # 3. Save: the JSON first, the PLY last, so that "PLY exists" means "both are complete".
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_bytes(json_bytes)
        glb.write_ply(ply_path, vertices, faces)
        row.update(status=DONE, num_vertices=len(vertices), num_faces=len(faces),
                   bytes=glb.HEADER_BYTES + json_length + (end - start), file_bytes=total_length)
    except Exception as error:  # an unsupported file, a missing file, or a network error after all retries
        row.update(status=FAILED, message=f"{type(error).__name__}: {error}")  # reported; the other meshes continue
    return row


def photo_file(folder: Path, photo: abo.AboImage) -> Path:
    """Return the local path of one photo: `<folder>/<class>/<image id><extension of the bucket file>`."""
    return folder / photo.category / f"{photo.image_id}{Path(photo.path).suffix}"


def download_photo(fetcher: Fetcher, photo: abo.AboImage, folder: Path, size: str) -> dict:
    """Download one photo unless it exists. Returns {"image_id", "status", "bytes", "message"}."""
    target = photo_file(folder, photo)
    if target.exists():
        return {"image_id": photo.image_id, "status": SKIPPED, "bytes": 0, "message": ""}
    try:
        return {"image_id": photo.image_id, "status": DONE, "bytes": fetcher.save(abo.image_url(photo.path, size), target), "message": ""}
    except Exception as error:  # reported; the other photos continue
        return {"image_id": photo.image_id, "status": FAILED, "bytes": 0, "message": f"{type(error).__name__}: {error}"}


def _run_parallel(function, items: list, workers: int, description: str) -> list[dict]:
    """Run `function(item)` for every item in `workers` threads, with a progress bar. Results keep the item order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(function, items), total=len(items), desc=description, unit="file"))


def _merge_mesh_rows(path: Path, new_rows: list[dict]) -> list[dict]:
    """Update `mesh_files.csv`: a skipped mesh keeps the row of the run that downloaded it."""
    old = {}
    if path.exists():
        with open(path, newline="") as f:
            old = {row["mesh_id"]: row for row in csv.DictReader(f)}
    merged = [old.get(r["mesh_id"], r) if r["status"] == SKIPPED else r for r in new_rows]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MESH_FILE_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    return merged


def run(cfg: dict, mini: bool = False, stages: tuple[str, ...] = STAGES, workers: int | None = None) -> dict:
    """Run the ABO download step, and return its report (also saved as JSON).

    Args:
        cfg: The merged settings; the step reads `cfg["data"]` (class rules, photo size, workers, retries).
        mini: True = 3 meshes per class and their photos, into test_outputs/download_abo_mini/.
        stages: Which of "metadata", "meshes", "photos" to run. The selection always runs.
        workers: Parallel downloads (None = `data.abo.download_workers`).
    """
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
    data_cfg = cfg["data"]
    abo_cfg = data_cfg["abo"]
    workers = workers or abo_cfg["download_workers"]
    out = output_dirs(mini)
    fetcher = Fetcher(retries=abo_cfg["retries"])
    report: dict = {"dataset": abo.DATASET, "mini": mini, "stages": list(stages), "workers": workers}
    start = time.perf_counter()

    # 1. Metadata.
    if "metadata" in stages:
        report["metadata"] = download_metadata(fetcher, out["metadata"])

    # 2. Selection and manifest.
    meshes, photos = select_from_metadata(out["metadata"], data_cfg, mini)
    abo.write_manifest(meshes, photos, out["manifest"])
    report["selection"] = {"meshes": len(meshes), "photos": len(photos), "classes": abo.class_counts(meshes, photos)}
    print(f"selection: {len(meshes)} meshes, {len(photos)} photos in {len(report['selection']['classes'])} classes")

    # 3. Mesh geometry.
    if "meshes" in stages:
        t0 = time.perf_counter()
        rows = _run_parallel(lambda m: download_mesh(fetcher, m, out["meshes"]), meshes, workers, "meshes")
        rows = _merge_mesh_rows(out["manifest"] / "mesh_files.csv", rows)
        status = Counter(r["status"] for r in rows)
        downloaded = sum(int(r["bytes"] or 0) for r in rows if r["status"] == DONE)
        full = sum(int(r["file_bytes"] or 0) for r in rows if r["status"] == DONE)
        report["meshes"] = {
            "status": dict(status),
            "bytes_downloaded": downloaded,
            "share_of_full_files": round(downloaded / full, 4) if full else None,
            "seconds": round(time.perf_counter() - t0, 1),
            "failed": [{"mesh_id": r["mesh_id"], "message": r["message"]} for r in rows if r["status"] == FAILED],
        }
        print(f"meshes: {dict(status)}; {downloaded / 1e6:.1f} MB")

    # 4. Photos.
    if "photos" in stages:
        t0 = time.perf_counter()
        results = _run_parallel(lambda p: download_photo(fetcher, p, out["photos"], abo_cfg["photos"]), photos, workers, "photos")
        status = Counter(r["status"] for r in results)
        failures = defaultdict(int)
        for r in results:
            if r["status"] == FAILED:
                failures[r["message"].split(":")[0]] += 1
        report["photos"] = {
            "status": dict(status),
            "bytes_downloaded": sum(r["bytes"] for r in results),
            "seconds": round(time.perf_counter() - t0, 1),
            "failure_kinds": dict(failures),
            "failed": [r["image_id"] for r in results if r["status"] == FAILED],
        }
        print(f"photos: {dict(status)}; {report['photos']['bytes_downloaded'] / 1e6:.1f} MB")

    report["seconds"] = round(time.perf_counter() - start, 1)
    report_path = out["manifest"] / "download_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {report_path.relative_to(paths.ROOT)} ({report['seconds']} s)")
    return report
