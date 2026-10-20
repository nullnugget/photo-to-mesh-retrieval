"""Train the three embedding heads on one split, and record the run.

Each epoch passes once over the training photos of the split (randomly changed query images through
the frozen backbone) against every mesh of the training gallery, then measures R@1 on the val
photos of the split (stored features, both mask sets for Pix3D) against the full gallery. The heads
of the best epoch are saved as weights/checkpoints/<run_name>/embedding_heads_best.pt; the log,
the metrics, and the settings go to runs/<run_name>/, and runs/registry.csv gets one row per val
query set.

A mini run (--mini: 5 % of the training photos, 1 epoch) writes into test_outputs/train_mini/.

Cross-validation (--folds): one training per fold of the split's non-test meshes (with the same
settings and tag), then the pooled validation over all non-test meshes (`src/pipeline/cross_validate.py`).
--folds all trains every fold; --folds 3 4 5 continues a stopped cross-validation.

The logic is in `src/pipeline/train_heads.py` and `src/train/trainer.py`.

Usage (from the repository root):
    python scripts/06_train.py --split seen_meshes                     # Pix3D (data.datasets), the settings of configs/train.yaml
    python scripts/06_train.py --split seen_meshes --mini              # quick check
    python scripts/06_train.py --split unseen_meshes --tag first
    python scripts/06_train.py --split seen_meshes --set train.loss.temperature=0.05 --tag tau005
    python scripts/06_train.py --split unseen_meshes --folds all --tag cvbase       # 5 trainings, then the pooled result
    python scripts/06_train.py --split unseen_categories --folds all --tag cvcat     # one training per category, then the pooled result
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import cross_validate, extract_features, train_heads  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=paths.SPLIT_NAMES, default="seen_meshes", help="split to train on")
    parser.add_argument("--datasets", nargs="+", choices=extract_features.DATASETS, help="datasets (default: data.datasets)")
    parser.add_argument("--backbone", help="backbone of the feature files (default: backbone.name)")
    parser.add_argument("--tag", help="extra word for the run name")
    parser.add_argument("--mini", action="store_true", help="5 %% of the training photos, 1 epoch, into test_outputs/")
    parser.add_argument("--folds", nargs="+", help="cross-validation: 'all' or fold numbers (needs --tag)")
    add_config_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    if args.folds:
        if not args.tag:
            parser.error("--folds needs --tag (the name of the cross-validation)")
        folds = None if args.folds == ["all"] else [int(k) for k in args.folds]
        cross_validate.run_folds(cfg, args.split, args.tag, folds=folds, datasets=args.datasets, backbone=args.backbone, mini=args.mini)
        return
    train_heads.run(cfg, split=args.split, datasets=args.datasets, backbone=args.backbone,
                    tag=args.tag, mini=args.mini)


if __name__ == "__main__":
    main()
