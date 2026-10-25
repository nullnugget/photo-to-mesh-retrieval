"""The photo check step (ABO): the bad score of every photo that the segment step kept.

Some ABO extra images show only a part of the object (a leg, a drawer pull, the fabric, a corner) or no
photo of it (a line drawing, a swatch). The segment step's box and mask rules cannot find them, so this
step asks SigLIP (`src/preprocess/photo_check.py`) whether each photo shows the whole object. The main
image of a listing always shows the whole product; it gets a score too, for the record, but the rule
uses only the extra images.

The rule (`segment.cleaning.max_bad_score`): an extra image with a bad score above it is left out by the
query step (`03_make_queries.py --dataset abo --keep-parts`), from every part. The threshold was chosen
on 444 reviewed extra images (06 P41): at 0.4, about 88 % of the close-ups go and 1.6 % of the good
photos.

Outputs:
    data/derived/masks/abo_photo_check.csv           photo_id, kind (main / extra), bad_score
    data/derived/masks/abo_photo_check_report.json   counts, the share above the threshold per class
"""

import csv
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src import paths
from src.data import abo
from src.data.catalog import read_table
from src.pipeline import download_abo, segment_photos
from src.preprocess.photo_check import PhotoCheck

MAIN, EXTRA = "main", "extra"
BATCH = 32


def results_file(root: Path = paths.MASKS_DIR) -> Path:
    return root / "abo_photo_check.csv"


def read_results(root: Path = paths.MASKS_DIR) -> dict[str, tuple[str, float]]:
    """Photo id -> (kind, bad score) of the photo check step."""
    return {r["photo_id"]: (r["kind"], float(r["bad_score"])) for r in read_table(results_file(root))}


def is_left_out(kind: str, bad_score: float, max_bad_score: float) -> bool:
    """The rule: an extra image whose bad score is above the threshold is left out; a main image never."""
    return kind == EXTRA and bad_score > max_bad_score


def _load(path: Path) -> Image.Image:
    """The full photo, as in the review that chose the threshold (a reduced JPEG decode would change the scores a little)."""
    with Image.open(path) as im:
        return im.convert("RGB")


def run(cfg: dict, device: str = "cuda") -> dict:
    """Score every kept ABO photo; write the results table and a report."""
    segment_cfg = cfg["segment"]
    start = time.perf_counter()
    kept = [r for r in read_table(segment_photos.results_file(abo.DATASET)) if r["verdict"] == segment_photos.KEPT]
    _, photos = abo.read_manifest(paths.ABO_MANIFEST_DIR)
    by_id = {p.image_id: p for p in photos}
    main = abo.main_image_ids(sorted(paths.ABO_METADATA_DIR.glob("listings_*.json.gz")))
    items = [(r["photo_id"], by_id[r["photo_id"].split("/")[-1]]) for r in kept]
    check = PhotoCheck(segment_cfg["photo_check"], segment_cfg.get("prompts"), device)
    rows = []
    with ThreadPoolExecutor(8) as pool:
        for i in tqdm(range(0, len(items), BATCH), desc="photo check", mininterval=10):
            batch = items[i:i + BATCH]
            images = list(pool.map(lambda item: _load(download_abo.photo_file(paths.ABO_IMAGES_DIR, item[1])), batch))
            scores = check.bad_scores(images, [photo.category for _, photo in batch])
            rows += [{"photo_id": photo_id, "kind": MAIN if photo.image_id in main else EXTRA, "bad_score": f"{s:.4f}"}
                     for (photo_id, photo), s in zip(batch, scores)]
    with open(results_file(), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo_id", "kind", "bad_score"])
        writer.writeheader()
        writer.writerows(rows)
    threshold = segment_cfg["cleaning"]["max_bad_score"]
    out = [r for r in rows if is_left_out(r["kind"], float(r["bad_score"]), threshold)]
    per_class = Counter(r["photo_id"].split("/")[1] for r in rows)
    out_class = Counter(r["photo_id"].split("/")[1] for r in out)
    report = {"photos": len(rows), "main": sum(r["kind"] == MAIN for r in rows), "extra": sum(r["kind"] == EXTRA for r in rows),
              "max_bad_score": threshold, "left_out": len(out),
              "left_out_per_class": {c: f"{out_class[c]} of {n}" for c, n in sorted(per_class.items())},
              "seconds": round(time.perf_counter() - start, 1)}
    (paths.MASKS_DIR / "abo_photo_check_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"photo check: {len(rows)} photos ({report['main']} main, {report['extra']} extra); "
          f"{len(out)} extra images above {threshold} -> {results_file().relative_to(paths.ROOT)} ({report['seconds']} s)")
    return report
