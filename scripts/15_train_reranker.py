"""Train and measure re-ranking method 2 (the learned pair scorer) by cross-validation.

One re-ranker per fold of the stage-1 cross-validation `--tag`: it trains on the out-of-fold lists of the
fold's training photos (only the fold's training meshes in the lists) and is measured on the fold's val
photos, against method 1 and stage 1. The variant (`rerank.method2.variant`) decides the base score:
residual_method1 (stage 1 + method 1 + correction) or stage1 (stage 1 + correction). Settings:
`configs/rerank.yaml` (`rerank.method2`); the patch store of `rerank.patches.size`.

A mini run (--mini) trains fold 1 for 1 epoch on a few hundred photos, into `test_outputs/train_reranker_mini/`.

The logic is in `src/pipeline/train_reranker.py`; the model in `src/models/reranker.py`.

Usage (from the repository root):
    python scripts/15_train_reranker.py --tag cvjointclean --mini
    python scripts/15_train_reranker.py --tag cvjointclean
    python scripts/15_train_reranker.py --tag cvjointclean --set rerank.method2.variant=stage1
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import train_reranker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="tag of the stage-1 cross-validation (its out-of-fold lists)")
    parser.add_argument("--datasets", nargs="+", default=["pix3d", "abo"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--mini", action="store_true", help="fold 1, 1 epoch, a few hundred photos, into test_outputs/")
    add_config_args(parser)
    args = parser.parse_args()
    train_reranker.run(config_from_args(args), args.tag, args.folds, args.datasets, mini=args.mini)


if __name__ == "__main__":
    main()
