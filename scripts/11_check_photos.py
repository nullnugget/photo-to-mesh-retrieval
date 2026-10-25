"""Check every kept ABO photo with SigLIP: does it show the whole object, or only a part, a drawing, a swatch?

Run after the segment step (02) and before the query step (03). The query step then leaves out the extra
images whose bad score is above `segment.cleaning.max_bad_score`; main images are always kept. The model
must be downloaded first (`00_download_weights.py --models photo_check`). About 5 minutes on the GPU.

The logic is in `src/pipeline/check_photos.py`; the model in `src/preprocess/photo_check.py`.

Usage (from the repository root):
    python scripts/11_check_photos.py
    python scripts/03_make_queries.py --dataset abo --keep-parts     # then: the photos above the threshold leave every part
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import check_photos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    add_config_args(parser)
    args = parser.parse_args()
    check_photos.run(config_from_args(args), args.device)


if __name__ == "__main__":
    main()
