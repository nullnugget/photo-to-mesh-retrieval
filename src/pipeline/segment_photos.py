"""The segment step: an object mask for every photo of one dataset, made with SAM 2.1.

Two ways to get the box that SAM needs (`prompt`):

- "reference_box": the box around the photo's reference mask (Pix3D has hand-made masks). This
  is the quality check of SAM (how close are SAM's masks to the reference masks?) and the input of
  the end-to-end test (retrieval with SAM masks, as in the demo, where the user draws the box).
- "detector": Grounding DINO finds the box from the class name (ABO photos, which have no masks).
  Then the cleaning rules decide whether the photo is kept (see `cleaning_verdict`). If SAM's mask is
  the white background of a product photo, one of SAM's three candidate masks that shows the object
  replaces it (see `is_background_mask`, `object_candidate`). ABO photos that show the meshes of
  several gallery entries are not segmented (see `abo_items`).

For every photo, the step:

1. reads the photo, and gets the box (from the reference mask, or the most confident detection);
2. makes the SAM mask from the box, and saves it in the Pix3D format (0 = background, 255 = object);
3. writes one row of results: the box, SAM's quality estimate, the mask area, and
   - with a reference mask: the IoU against it, plain and edge-tolerant (see `segment.edge_band`);
   - with the detector: its confidence and the cleaning verdict (kept or removed, and why).

Which photos (Pix3D): all photos of the catalog, or only some parts of a split, for example the
train and val parts of seen_meshes (the 8,557 photos that may be used for tuning; the test photos
are kept for the final result). ABO: all photos of the download manifest.

Safe to stop and start again: masks that exist are skipped (and keep their earlier results row).
A photo without a detection has no mask file, so a new run asks the detector again (it is quick).
The settings are saved in `<dataset>_segment_config.yaml`; a run with other settings stops with an
error instead of mixing masks, unless `overwrite=True` (then every mask is made again).

Outputs (normal run):
    data/derived/masks/<dataset>/<category>/<name>.png      the masks
    data/derived/masks/<dataset>_segment_config.yaml        the settings that made them
    data/derived/masks/<dataset>_segment_results.csv        one row per photo (see RESULT_COLUMNS)
    data/derived/masks/<dataset>_segment_report.json        totals of all rows

A mini run segments 3 random photos of each category (or `mini_count`; fixed seed) into `test_outputs/segment_mini/`.
"""

import csv
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src import paths
from src.config import load_yaml, save_config
from src.data import abo
from src.data.catalog import read_table
from src.data.splits import load_split
from src.pipeline import catalog_abo_meshes, download_abo
from src.preprocess import segment

PROMPTS = ("reference_box", "detector")
MINI_PHOTOS_PER_CATEGORY = 3
MINI_SEED = 0  # seed of the random photos of a mini run
DONE, SKIPPED, FAILED = "done", "skipped", "failed"
KEPT, REMOVED = "kept", "removed"
# One row per photo in <dataset>_segment_results.csv. Cells that do not apply stay empty (for
# example the IoU columns without a reference mask, or the detector columns with a reference box).
RESULT_COLUMNS = [
    "photo_id",  # "pix3d/chair/0001" (the query id) or "abo/<class>/<image id>"
    "category",
    "status",  # done, skipped (mask existed), failed (see message)
    "verdict",  # detector only: kept or removed
    "reason",  # why the photo was removed: no_detection, background_mask, close_up, box_too_small, mask_too_small,
    #            mask_too_large
    "box",  # "x0 y0 x1 y1" in photo pixels
    "detector_score",  # confidence of the box
    "edges_touched",  # photo edges that the box touches (0 to 4)
    "sam_score",  # SAM's own estimate of the mask quality
    "mask_area",  # share of the photo covered by the mask
    "mask_repaired",  # detector only: 1 if SAM's mask was the background and a candidate replaced it
    "iou",  # against the reference mask
    "iou_edge_tolerant",  # the same, without the pixels near the reference edge
    "message",
]


class SegmentSettingsChanged(RuntimeError):
    """The existing masks were made with other settings (see `<dataset>_segment_config.yaml`)."""


@dataclass
class PhotoItem:
    """One photo to segment."""

    photo_id: str
    category: str
    photo_path: str  # relative to the repository root
    reference_mask_path: str | None  # the dataset's own mask, or None
    mask_name: str  # file name of the SAM mask inside <masks>/<dataset>/<category>/


def output_root(mini: bool) -> Path:
    """Return the folder of the masks: the real one, or a test folder for a mini run."""
    return paths.TEST_OUTPUTS_DIR / "segment_mini" / "masks" if mini else paths.MASKS_DIR


def mask_file(root: Path, dataset: str, item: PhotoItem) -> Path:
    """Return the SAM mask file of one photo, for example <root>/pix3d/chair/0001.png."""
    return root / dataset / item.category / item.mask_name


def results_file(dataset: str, root: Path = paths.MASKS_DIR) -> Path:
    return root / f"{dataset}_segment_results.csv"


def report_file(dataset: str, root: Path = paths.MASKS_DIR) -> Path:
    return root / f"{dataset}_segment_report.json"


def config_file(dataset: str, root: Path = paths.MASKS_DIR) -> Path:
    return root / f"{dataset}_segment_config.yaml"


# ---------------------------------------------------------------------------------------------
# The photos of each dataset
# ---------------------------------------------------------------------------------------------


def pix3d_items(split: str | None = None, parts: list[str] | None = None) -> list[PhotoItem]:
    """The Pix3D photos of the catalog; with `split` and `parts`, only the photos of those parts."""
    rows = [r for r in read_table(paths.QUERIES_CSV) if r["dataset"] == "pix3d"]
    if split:
        split_parts = load_split(paths.split_file("pix3d", split))["parts"]
        wanted = {query_id for part in parts or ["train", "val", "test"] for query_id in split_parts[part]}
        rows = [r for r in rows if r["query_id"] in wanted]
    return [
        PhotoItem(r["query_id"], r["category"], r["photo_path"], r["mask_path"], Path(r["photo_path"]).stem + ".png")
        for r in rows
    ]


def abo_items() -> list[PhotoItem]:
    """The ABO photos of the download manifest that show the meshes of exactly one gallery entry.

    A photo that the listings of several different geometries share (a family photo of a product
    line, for example) has no single correct answer, so the query step leaves it out; this step does
    not segment it either (see `catalog_abo_meshes.photo_entry`). ABO photos have no reference masks.
    """
    _, photos = abo.read_manifest(paths.ABO_MANIFEST_DIR)
    groups = catalog_abo_meshes.read_geometry_groups(paths.ABO_MANIFEST_DIR)
    usable = [p for p in photos if catalog_abo_meshes.photo_entry(p, groups) is not None]
    print(f"abo: {len(photos) - len(usable)} of {len(photos)} photos show meshes of several gallery entries; not segmented")
    return [
        PhotoItem(
            f"{abo.DATASET}/{p.category}/{p.image_id}", p.category,
            str(download_abo.photo_file(paths.ABO_IMAGES_DIR, p).relative_to(paths.ROOT)), None, f"{p.image_id}.png",
        )
        for p in usable
    ]


def _sample_per_category(items: list[PhotoItem], count: int, seed: int = MINI_SEED) -> list[PhotoItem]:
    """Keep `count` random photos of each category (fixed seed), in their original order.

    Random, and not the first photos: the ABO photos are sorted by image id, and the first ids of a
    class are mostly small images (often colour swatches), so the first photos are not typical.
    """
    by_category: dict[str, list[PhotoItem]] = {}
    for item in items:
        by_category.setdefault(item.category, []).append(item)
    rng = random.Random(seed)
    chosen = {id(i) for group in by_category.values() for i in rng.sample(group, min(count, len(group)))}
    return [i for i in items if id(i) in chosen]


# ---------------------------------------------------------------------------------------------
# One photo
# ---------------------------------------------------------------------------------------------


def cleaning_verdict(
    box: tuple[float, float, float, float], width: int, height: int, mask_area: float, cleaning_cfg: dict
) -> tuple[str, str, str]:
    """Apply the cleaning rules to a detected photo; return (verdict, reason code, detail).

    The rules (`segment.cleaning`; the detector threshold is applied before), in this order:
    - close_up: the box touches more than `max_edges_touched` photo edges, or it touches any edge and
      covers at least `close_up_box_share` of the photo. A crop of a detail fills the frame; an object
      that the frame of a room photo cuts (a bed at the side of the photo) has a smaller box.
    - box_too_small: the longer box side is under `min_box_side` of the longer photo side, or under
      `min_box_pixels`. The box and not the mask is measured, because a thin object (a floor lamp)
      spans the photo while its mask covers only a few per cent.
    - mask_too_small, mask_too_large: the mask covers less than `min_mask_area` or more than
      `max_mask_area` of the photo (SAM found almost nothing, or a swatch fills the frame).

    Args:
        box: The detector box (x0, y0, x1, y1), in photo pixels.
        width, height: The photo size in pixels.
        mask_area: The share of the photo that the mask covers.
        cleaning_cfg: `segment.cleaning`.
    """
    edges = segment.edges_touched(box, width, height)
    x0, y0, x1, y1 = box
    box_share = (x1 - x0) * (y1 - y0) / (width * height)
    box_side = max(x1 - x0, y1 - y0)
    if edges > cleaning_cfg["max_edges_touched"]:
        return REMOVED, "close_up", f"the box touches {edges} photo edges"
    if edges >= 1 and box_share >= cleaning_cfg["close_up_box_share"]:
        return REMOVED, "close_up", f"the box touches {edges} photo edge(s) and covers {box_share:.2f} of the photo"
    if box_side < cleaning_cfg["min_box_side"] * max(width, height) or box_side < cleaning_cfg["min_box_pixels"]:
        return REMOVED, "box_too_small", f"longer box side {box_side:.0f} pixels of {max(width, height)}"
    if mask_area < cleaning_cfg["min_mask_area"]:
        return REMOVED, "mask_too_small", f"mask area {mask_area:.3f}"
    if mask_area > cleaning_cfg["max_mask_area"]:
        return REMOVED, "mask_too_large", f"mask area {mask_area:.3f}"
    return KEPT, "", ""


def is_background_mask(white_share: float, coverage: float, background_cfg: dict) -> bool:
    """True if a mask is the white background of a product photo instead of the object.

    The mask is mostly near-white (`white_share_above`) and covers little of the coloured pixels in
    the box (`coverage_below`); see `segment.background_measures`.
    """
    return white_share >= background_cfg["white_share_above"] and coverage < background_cfg["coverage_below"]


def object_candidate(image: np.ndarray, box, candidates: list, background_cfg: dict):
    """Return the SAM candidate that shows the object, or None if no candidate does.

    A candidate shows the object if its white share is below `repair_white_share_below` and it covers
    at least `repair_coverage_above` of the coloured pixels in the box. Of those, the candidate with the
    highest coverage wins. SAM's own scores are not used: SAM often rates a clean mask of the white
    background higher than the mask of the object.
    """
    best, best_coverage = None, -1.0
    for candidate in candidates:
        white_share, coverage = segment.background_measures(image, candidate.mask, box, background_cfg["near_white_level"])
        if white_share < background_cfg["repair_white_share_below"] and coverage >= background_cfg["repair_coverage_above"]:
            if coverage > best_coverage:
                best, best_coverage = candidate, coverage
    return best


def segment_photo(item: PhotoItem, target: Path, segmenter, detector, segment_cfg: dict, prompt: str) -> dict:
    """Make and save the mask of one photo; return its results row (see RESULT_COLUMNS)."""
    row = {column: "" for column in RESULT_COLUMNS}
    row.update(photo_id=item.photo_id, category=item.category, status=DONE)
    with Image.open(paths.resolve(item.photo_path)) as photo_file:
        image = np.asarray(photo_file.convert("RGB"))
    height, width = image.shape[:2]

    # 1. The box.
    reference = None
    if prompt == "reference_box":
        reference = segment.read_mask(item.reference_mask_path)
        box = segment.box_of_mask(reference)
    else:
        detections = detector.detect(image, segment.prompt_text(item.category, segment_cfg.get("prompts")))
        if not detections:
            row.update(verdict=REMOVED, reason="no_detection")
            return row
        box = detections[0].box  # the most confident detection
        row["detector_score"] = round(detections[0].score, 4)
    edges = segment.edges_touched(box, width, height)
    row.update(box=" ".join(f"{v:.1f}" for v in box), edges_touched=edges)

    # 2. The mask. For a detected photo, SAM also returns its three candidates: if its mask is the
    # white background of a product photo, a candidate that shows the object replaces it.
    background = False
    if prompt == "reference_box":
        result = segmenter.mask_from_box(image, box)
    else:
        result, candidates = segmenter.mask_with_candidates(image, box)
        background_cfg = segment_cfg["cleaning"]["background"]
        if is_background_mask(*segment.background_measures(image, result.mask, box, background_cfg["near_white_level"]), background_cfg):
            repaired = object_candidate(image, box, candidates, background_cfg)
            if repaired is None:
                background = True  # removed below; the background mask is saved for the review
            else:
                result = repaired
                row["mask_repaired"] = 1
    segment.save_mask(result.mask, target)
    area = float(result.mask.mean())
    row.update(sam_score=round(result.score, 4), mask_area=round(area, 4))

    # 3. Comparison with the reference mask, or the cleaning verdict.
    if reference is not None:
        band = segment.edge_band(reference, segment.edge_band_width(reference, segment_cfg["check"]["edge_band_fraction"]))
        row.update(iou=round(segment.mask_iou(result.mask, reference), 4),
                   iou_edge_tolerant=round(segment.mask_iou(result.mask, reference, ignore=band), 4))
    elif background:
        row.update(verdict=REMOVED, reason="background_mask", message="SAM's mask and its candidates show the background")
    else:
        verdict, reason, detail = cleaning_verdict(box, width, height, area, segment_cfg["cleaning"])
        row.update(verdict=verdict, reason=reason, message=detail)
    return row


# ---------------------------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------------------------


def _check_settings(record: Path, settings: dict, overwrite: bool) -> None:
    """Stop if earlier masks were made with other settings; save the current settings."""
    if record.exists() and not overwrite and load_yaml(record) != settings:
        raise SegmentSettingsChanged(
            f"{record} differs from the current settings. Run again with overwrite (all masks are made "
            "again), or restore the settings."
        )
    record.parent.mkdir(parents=True, exist_ok=True)
    save_config(settings, record)


def _merge_results(path: Path, new_rows: list[dict]) -> list[dict]:
    """Update the results table: a skipped photo keeps its earlier row; other photos get the new row."""
    old = {}
    if path.exists():
        with open(path, newline="") as f:
            old = {row["photo_id"]: row for row in csv.DictReader(f)}
    merged = [old.get(row["photo_id"], row) if row["status"] == SKIPPED else row for row in new_rows]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    return merged


def _summary(rows: list[dict]) -> dict:
    """Totals of the results rows: IoU statistics (reference box) or the cleaning verdicts (detector)."""
    summary: dict = {"status": dict(Counter(r["status"] for r in rows))}
    for column in ("iou", "iou_edge_tolerant"):
        values = np.array([float(r[column]) for r in rows if r[column] not in ("", None)])
        if values.size:
            summary[column] = {
                "median": round(float(np.median(values)), 4),
                "mean": round(float(values.mean()), 4),
                "share_below_0.70": round(float((values < 0.70).mean()), 4),
                "share_below_0.90": round(float((values < 0.90).mean()), 4),
            }
    verdicts = Counter(r["verdict"] for r in rows if r["verdict"])
    if verdicts:
        summary["verdicts"] = dict(verdicts)
        summary["masks_repaired"] = sum(str(r.get("mask_repaired", "")) == "1" for r in rows)
        summary["removed_because"] = dict(Counter(r["reason"] for r in rows if r["verdict"] == REMOVED))
    return summary


def run(
    dataset: str,
    cfg: dict,
    prompt: str | None = None,
    split: str | None = None,
    parts: list[str] | None = None,
    mini: bool = False,
    overwrite: bool = False,
    device: str = "cuda",
    mini_count: int = MINI_PHOTOS_PER_CATEGORY,
) -> dict:
    """Segment the photos of one dataset, and return the report (also saved as JSON).

    Args:
        dataset: "pix3d" or "abo".
        cfg: The merged settings; the step reads `cfg["segment"]`.
        prompt: "reference_box" or "detector" (None = reference_box for Pix3D, detector for ABO).
        split, parts: Pix3D only: segment only these parts of this split (for example "seen_meshes",
            ["train", "val"]). None = all photos.
        mini: True = only `mini_count` random photos of each category (fixed seed), into test_outputs/segment_mini/.
        overwrite: True = make every mask again, also with changed settings.
        device: "cuda" or "cpu".
        mini_count: Photos per category of a mini run (3 by default; for example 20 for a sample
            that is large enough to review the ABO cleaning rules).

    Raises:
        ValueError: For an unknown dataset or prompt, a reference box without reference masks, or a
            split for ABO.
        SegmentSettingsChanged: If earlier masks used other settings and `overwrite` is False.
    """
    prompt = prompt or ("reference_box" if dataset == "pix3d" else "detector")
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")
    if dataset == "pix3d":
        items = pix3d_items(split, parts)
    elif dataset == "abo":
        if split:
            raise ValueError("ABO has no split files yet; segment all ABO photos")
        if prompt == "reference_box":
            raise ValueError("ABO photos have no reference masks; use the detector")
        items = abo_items()
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if mini:
        items = _sample_per_category(items, mini_count)

    segment_cfg = cfg["segment"]
    root = output_root(mini)
    _check_settings(config_file(dataset, root), {"prompt": prompt, "segment": segment_cfg}, overwrite)
    segmenter = segment.Segmenter(segment_cfg["segmenter"], device=device)
    detector = segment.Detector(segment_cfg["detector"], device=device) if prompt == "detector" else None
    print(f"{dataset}: {len(items)} photos{f' (mini, {mini_count} per category)' if mini else ''}; box from {prompt}; "
          f"SAM mask '{segmenter.box_mask}' -> {(root / dataset).relative_to(paths.ROOT)}")

    rows = []
    start = time.perf_counter()
    for item in tqdm(items, desc="segment", unit="photo"):
        target = mask_file(root, dataset, item)
        if target.exists() and not overwrite:
            rows.append({**{c: "" for c in RESULT_COLUMNS}, "photo_id": item.photo_id, "category": item.category, "status": SKIPPED})
            continue
        try:
            rows.append(segment_photo(item, target, segmenter, detector, segment_cfg, prompt))
        except (OSError, ValueError, RuntimeError) as error:  # an unreadable photo, an empty reference mask, out of memory
            rows.append({**{c: "" for c in RESULT_COLUMNS}, "photo_id": item.photo_id, "category": item.category,
                         "status": FAILED, "message": f"{type(error).__name__}: {error}"})

    this_run = Counter(row["status"] for row in rows)
    all_rows = _merge_results(results_file(dataset, root), rows)
    report = {
        "dataset": dataset,
        "mini": mini,
        "mini_count": mini_count if mini else None,
        "prompt": prompt,
        "split": split,
        "parts": parts,
        "box_mask": segmenter.box_mask,
        "photos": len(items),
        "this_run": dict(this_run),
        **_summary(all_rows),
        "failed": [{"photo_id": r["photo_id"], "message": r["message"]} for r in all_rows if r["status"] == FAILED],
        "seconds_this_run": round(time.perf_counter() - start, 1),
    }
    path = report_file(dataset, root)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"segment: this run {dict(this_run)}; "
          + (f"IoU median {report['iou']['median']}, edge-tolerant {report['iou_edge_tolerant']['median']}; " if "iou" in report else "")
          + (f"verdicts {report['verdicts']}; " if "verdicts" in report else "")
          + f"{report['seconds_this_run']} s -> {path.relative_to(paths.ROOT)}")
    return report
