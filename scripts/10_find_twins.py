"""Find near-identical meshes ("twins") of a dataset by geometry: grids, grid pairs, exact distances.

Many ABO meshes are colour variants of one form with slightly different files; their grey renders
are the same, so a search cannot tell them apart. This step measures the pairs by geometry only (no
backbone features); the last stage merges the twins into one gallery entry each. The method is in
`src/data/mesh_twins.py`, the stages in `src/pipeline/find_twins.py`.

Usage (from the repository root):
    python scripts/10_find_twins.py --dataset abo --stage grids                 # one grid per mesh (CPU, minutes)
    python scripts/10_find_twins.py --dataset abo --stage pairs --min-iou 0.5   # IoU of all pairs (GPU, seconds)
    python scripts/10_find_twins.py --dataset abo --stage distances --min-iou 0.8   # exact test (GPU, ~0.3 s per pair; continues)
    python scripts/10_find_twins.py --dataset abo --stage merge                 # one entry per twin group (data.abo.twins)
    python scripts/03_make_queries.py --dataset abo --keep-parts                # then: photos → new entries, splits kept
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import find_twins  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="abo", choices=["abo", "pix3d"])
    parser.add_argument("--stage", required=True, choices=["grids", "pairs", "distances", "merge"])
    parser.add_argument("--min-iou", type=float, default=0.5, help="pairs: lowest IoU kept; distances: lowest IoU measured")
    parser.add_argument("--device", default="cuda")
    add_config_args(parser)
    args = parser.parse_args()
    if args.stage == "grids":
        find_twins.build_grids(args.dataset)
    elif args.stage == "pairs":
        find_twins.grid_pairs(args.dataset, args.min_iou, args.device)
    elif args.stage == "distances":
        find_twins.distances(args.dataset, args.min_iou, args.device)
    else:
        find_twins.merge(config_from_args(args)["data"]["abo"]["twins"], args.dataset)


if __name__ == "__main__":
    main()
