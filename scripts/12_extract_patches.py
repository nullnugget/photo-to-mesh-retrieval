"""Extract the reduced patch vectors of every render and query image (the patch store for re-ranking).

Stage "fit" fits the PCA on the object patches of random non-test images; stage "extract" writes
every set of both datasets with `rerank.patches.dim` numbers per patch. The rows follow the feature
files, so the feature step must be complete. A mini run (--mini) writes a small sample into
`test_outputs/extract_patches_mini/`.

The logic is in `src/pipeline/extract_patches.py`; the store format in `src/models/patch_store.py`.

Usage (from the repository root):
    python scripts/12_extract_patches.py                         # fit (if missing), then extract every set
    python scripts/12_extract_patches.py --mini                  # quick check
    python scripts/12_extract_patches.py --stages fit --overwrite
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import extract_patches  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stages", nargs="+", choices=("fit", "extract"), default=["fit", "extract"])
    parser.add_argument("--backbone", help="backbone name from configs/backbone.yaml (default: backbone.name)")
    parser.add_argument("--mini", action="store_true", help="a small sample into test_outputs/")
    parser.add_argument("--overwrite", action="store_true", help="fit or write again if the files exist")
    add_config_args(parser)
    args = parser.parse_args()
    extract_patches.run(config_from_args(args), args.stages, backbone=args.backbone, mini=args.mini, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
