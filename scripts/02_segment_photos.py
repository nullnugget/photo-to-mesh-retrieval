"""Make an object mask for every photo of one dataset with SAM 2.1 (and Grounding DINO for ABO).

Pix3D: SAM gets the box around the photo's own (hand-made) mask. The results table compares each
SAM mask with that mask (IoU), so this is the quality check of SAM; the masks are also the input of
the end-to-end test. ABO: Grounding DINO finds the box from the class name, SAM makes the mask, and
the cleaning rules in `configs/segment.yaml` decide whether the photo is kept.

The masks have the Pix3D format (0 = background, 255 = object) and go to
`data/derived/masks/<dataset>/<category>/`. The step can be stopped and started again: existing
masks are skipped. With changed settings it stops, unless --overwrite is given.

A mini run (--mini) segments 3 random photos of each category (or --mini-count N; fixed seed) into
`test_outputs/segment_mini/`. The check notebooks show the results: `notebooks/05_segment_check.py`
for Pix3D (it can also start a mini run with a button), `notebooks/03_abo_check.py` for ABO.

The logic is in `src/pipeline/segment_photos.py`; the models in `src/preprocess/segment.py`.

Usage (from the repository root):
    python scripts/02_segment_photos.py --dataset pix3d --split seen_meshes --parts train val   # tuning photos
    python scripts/02_segment_photos.py --dataset pix3d --split seen_meshes --parts test        # final test photos
    python scripts/02_segment_photos.py --dataset pix3d --mini
    python scripts/02_segment_photos.py --dataset abo --mini --mini-count 20                    # sample to review the rules
    python scripts/02_segment_photos.py --dataset abo
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import segment_photos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=["pix3d", "abo"], default="pix3d", help="dataset to segment")
    parser.add_argument("--prompt", choices=segment_photos.PROMPTS, default=None,
                        help="where the box comes from (default: reference_box for pix3d, detector for abo)")
    parser.add_argument("--split", choices=paths.SPLIT_NAMES, default=None, help="pix3d: only photos of this split ...")
    parser.add_argument("--parts", nargs="+", choices=["train", "val", "test"], default=None, help="... and these parts")
    parser.add_argument("--mini", action="store_true", help="only a few photos per category, into test_outputs/")
    parser.add_argument("--mini-count", type=int, default=segment_photos.MINI_PHOTOS_PER_CATEGORY,
                        help="photos per category of a mini run (default 3)")
    parser.add_argument("--overwrite", action="store_true", help="make every mask again (also with changed settings)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    add_config_args(parser)
    args = parser.parse_args()
    segment_photos.run(args.dataset, config_from_args(args), prompt=args.prompt, split=args.split, parts=args.parts,
                       mini=args.mini, overwrite=args.overwrite, device=args.device,
                       mini_count=args.mini_count)


if __name__ == "__main__":
    main()
