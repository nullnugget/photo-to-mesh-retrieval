"""Make the query images, the catalog rows, and the split files of one dataset.

A query image is a photo after preparation: the background is replaced with neutral grey, the
image is cropped tightly around the object, padded to a square, and resized to 224 x 224.

Outputs (normal run):
    data/derived/queries/<dataset>/<category>/<name>.png    query images
    data/derived/catalog/queries.csv, meshes.csv            catalog (the rows of this dataset replaced)
    data/derived/catalog/<dataset>_queries_report.json      counts, left-out photos, settings
    data/splits/<dataset>/<split name>.json                 seen_meshes, unseen_meshes, unseen_categories

Existing query images are kept unless --overwrite is given; the catalog rows and the split files
are always written again. A mini run (--mini) prepares 5 photos per category and writes everything
to `test_outputs/make_queries_mini/`. The check notebook `notebooks/01_query_check.py` can start a
mini run with a button and then shows the results.

The logic is in `src/pipeline/make_queries.py`.

Usage (from the repository root):
    python scripts/03_make_queries.py --dataset pix3d               # all photos, about 10 minutes
    python scripts/03_make_queries.py --dataset pix3d --mini        # quick check, a few seconds
    python scripts/03_make_queries.py --dataset pix3d --overwrite   # make the query images again
    python scripts/03_make_queries.py --dataset pix3d --set data.crop_margin=0.05
    python scripts/03_make_queries.py --dataset abo                 # after the ABO segment step (kept photos)
    python scripts/03_make_queries.py --dataset abo --mini          # from the segment step's mini outputs
    python scripts/03_make_queries.py --dataset abo --keep-parts    # after merged ABO entries: photos keep their parts
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import make_queries  # noqa: E402
from src.preprocess.queries import default_workers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=make_queries.DATASETS, default="pix3d", help="dataset to prepare")
    parser.add_argument("--mini", action="store_true", help="5 photos per category, into test_outputs/")
    parser.add_argument("--overwrite", action="store_true", help="make the query images again if they exist")
    parser.add_argument("--workers", type=int, default=default_workers(), help="parallel processes")
    parser.add_argument("--keep-parts", action="store_true", help="each photo keeps its part of the existing split files (after a catalog change)")
    add_config_args(parser)
    args = parser.parse_args()
    make_queries.run(args.dataset, config_from_args(args), mini=args.mini, overwrite=args.overwrite, workers=args.workers, keep_parts=args.keep_parts)


if __name__ == "__main__":
    main()
