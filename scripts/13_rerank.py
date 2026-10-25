"""Re-ranking (stage 2), measured on the folds of a stage-1 cross-validation.

Stages: "candidates" (the out-of-fold stage-1 lists of every val photo, GPU), "method1" (the
patch-by-patch scores of the first candidates, from the patch store of `12_extract_patches.py`, GPU),
"report" (R@1 after the re-ordering, the weight chosen by leaving one fold out, the paired difference
against stage 1), "gallery" (the patch comparison against every mesh, for a sample of photos, as a
stage 1), "dimcheck" (method 1 with the raw 768 numbers against the stored reduction, for a sample),
"sizecheck" (method 1 with the patches of larger images, 224 / 336 / 448 pixels, for a sample; it needs the
448-pixel images of `04_render.py --set render.output_size=448` and `14_query_images_at_size.py --size 448`;
these three are checks, not in the default stages). Settings: `configs/rerank.yaml`.

The logic is in `src/pipeline/rerank.py`.

Usage (from the repository root):
    python scripts/13_rerank.py --tag cvjointclean                              # all stages
    python scripts/13_rerank.py --tag cvjointclean --stages report
    python scripts/13_rerank.py --tag cvjointclean --stages gallery dimcheck
    python scripts/13_rerank.py --tag cvjointclean --stages sizecheck
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import rerank  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="tag of the stage-1 cross-validation (its fold models make the lists)")
    parser.add_argument("--split", choices=paths.SPLIT_NAMES, default="unseen_meshes")
    parser.add_argument("--datasets", nargs="+", default=["pix3d", "abo"], help="datasets of the stage-1 trainings")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stages", nargs="+", choices=("candidates", "method1", "report", "gallery", "dimcheck", "sizecheck"), default=["candidates", "method1", "report"])
    add_config_args(parser)
    args = parser.parse_args()
    rerank.run(config_from_args(args), args.stages, args.split, args.tag, args.datasets, args.folds)


if __name__ == "__main__":
    main()
