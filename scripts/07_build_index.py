"""Embed the gallery of a trained checkpoint once, save it as the gallery index, and check it.

The heads do not change after training, so the view embeddings and the mesh embeddings of every
gallery mesh are calculated once. The index file goes next to the checkpoint:
weights/checkpoints/<run_name>/embedding_heads_<best|last>_gallery_<dataset>.npz.

The check: the index searches the val photos of the training split, and its R@1 must equal the
validation R@1 that the training logged for the epoch of the checkpoint.

The logic is in `src/pipeline/build_index.py` and `src/retrieval/index.py`.

Usage (from the repository root):
    python scripts/07_build_index.py --run 2026-10-20_pix3d_dinov3_vitb16_seen_meshes_mix05
    python scripts/07_build_index.py --run <run_name> --which last
    python scripts/07_build_index.py --run <run_name of a mini training> --mini
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import build_index, extract_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="training run name (folder in weights/checkpoints/)")
    parser.add_argument("--which", choices=build_index.CHECKPOINT_KINDS, default="best", help="checkpoint of the run")
    parser.add_argument("--datasets", nargs="+", choices=extract_features.DATASETS, help="gallery datasets (default: those of the training)")
    parser.add_argument("--mini", action="store_true", help="a checkpoint of a mini training (test_outputs/train_mini/)")
    args = parser.parse_args()
    result = build_index.run(args.run, datasets=args.datasets, which=args.which, mini=args.mini)
    if not all(check["equal"] for check in result["checks"].values()):
        sys.exit("the index R@1 differs from the training log; see the lines above")


if __name__ == "__main__":
    main()
