"""Download the pretrained detector (Grounding DINO tiny) and segmenter (SAM 2.1 large).

The files and their fixed versions are listed in `configs/segment.yaml`; they go into
`weights/grounding_dino/grounding-dino-tiny/` and `weights/sam2/sam2.1-hiera-large/` (about 1.6 GB).
SAM 2.1 small (0.2 GB) is downloaded only on request, for comparisons and fast tests; SigLIP base (0.8 GB,
`weights/siglip/siglip-base-patch16-224/`) only on request, for the ABO photo check.
Files that exist with the right size are skipped. The DINOv3 and DINOv2 backbones are placed by
hand (see the README).

The logic is in `src/pipeline/download_weights.py`.

Usage (from the repository root):
    python scripts/00_download_weights.py                    # both models
    python scripts/00_download_weights.py --models segmenter        # only SAM 2.1 large
    python scripts/00_download_weights.py --models segmenter_small  # SAM 2.1 small, for comparisons and tests
    python scripts/00_download_weights.py --models photo_check      # SigLIP base, for the ABO photo check
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import download_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", choices=download_weights.MODELS, default=list(download_weights.DEFAULT_MODELS), help="models to download")
    add_config_args(parser)
    args = parser.parse_args()
    download_weights.run(config_from_args(args), models=tuple(args.models))


if __name__ == "__main__":
    main()
