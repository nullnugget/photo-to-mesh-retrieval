"""Extract the backbone features of the renders and the query images of one dataset.

For each feature set of the dataset (renders; query images with the dataset's own masks, Pix3D
only; query images with SAM masks), the step writes one feature file with both basic parts (class
token and masked mean) of every image. For Pix3D, the step first makes the query images with the
SAM masks. A set whose file exists is skipped, unless --overwrite is given.

A mini run (--mini) takes 2 meshes per category and 3 photos of each, into
`test_outputs/extract_features_mini/`. The check notebook `notebooks/07_feature_check.py` can start a
mini run with a button and checks the files.

The logic is in `src/pipeline/extract_features.py`.

Usage (from the repository root):
    python scripts/05_extract_features.py --dataset pix3d                     # all Pix3D sets
    python scripts/05_extract_features.py --dataset abo
    python scripts/05_extract_features.py --dataset pix3d --mini              # quick check
    python scripts/05_extract_features.py --dataset pix3d --feature-sets queries_sam --overwrite
    python scripts/05_extract_features.py --dataset pix3d --backbone dinov2_vitb14_reg
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import extract_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=extract_features.DATASETS, default="pix3d", help="dataset to extract")
    parser.add_argument("--feature-sets", nargs="+", choices=paths.FEATURE_SETS, help="sets to write (default: all sets of the dataset)")
    parser.add_argument("--backbone", help="backbone name from configs/backbone.yaml (default: backbone.name)")
    parser.add_argument("--mini", action="store_true", help="2 meshes per category, 3 photos each, into test_outputs/")
    parser.add_argument("--overwrite", action="store_true", help="write the sets again if their files exist")
    add_config_args(parser)
    args = parser.parse_args()
    extract_features.run(args.dataset, config_from_args(args), backbone=args.backbone, sets=args.feature_sets,
                         mini=args.mini, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
