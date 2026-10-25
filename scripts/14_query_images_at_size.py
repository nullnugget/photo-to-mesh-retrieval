"""Make the query images of every query set again at another side length (for example 448 pixels).

The images come from the original photos and the same masks, with the same crop as the main query
images; only the last resize differs. They go into folders of their own (`data/derived/queries_448/`,
`data/derived/masks/pix3d_sam_queries_448/`); the catalog, the splits, and the main images are not
touched. The renders of the same size come from the render step:
`04_render.py --dataset <dataset> --set render.output_size=448` (into `data/derived/renders_448/`).

A mini run (--mini) makes the first 16 images of each set into `test_outputs/sized_queries_mini/`.

The logic is in `src/pipeline/sized_queries.py`.

Usage (from the repository root):
    python scripts/14_query_images_at_size.py --size 448
    python scripts/14_query_images_at_size.py --size 448 --mini
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import sized_queries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", type=int, required=True, help="side length of the query images, in pixels")
    parser.add_argument("--datasets", nargs="+", default=["pix3d", "abo"])
    parser.add_argument("--mini", action="store_true", help="the first 16 images of each set, into test_outputs/")
    parser.add_argument("--overwrite", action="store_true", help="make existing images again")
    parser.add_argument("--workers", type=int, help="worker processes (default: up to 8)")
    add_config_args(parser)
    args = parser.parse_args()
    sized_queries.run(config_from_args(args), args.size, args.datasets, mini=args.mini, overwrite=args.overwrite,
                      workers=args.workers)


if __name__ == "__main__":
    main()
