"""Pool a finished cross-validation again, or compare two cross-validations.

A cross-validation trains one setting on every fold of a split's non-test meshes
(`python scripts/06_train.py --split unseen_meshes --folds all --tag <name>`); that command pools
the folds at its end. This script:

- pools the fold runs of a tag again (for example after a stopped run was continued in parts);
- compares two cross-validations (two settings) on the same val photos: the paired difference
  of R@1 over meshes, for each val query set and for all of them together. The two can have
  different datasets (Pix3D-only and Pix3D + ABO); the val query sets that both have are compared;
- compares two single trainings (no folds; for example seen_meshes) in the same way, from their
  per-photo val records.

--epoch last|best pools at that epoch (default: the rule `train.final_epoch` of the trainings for
--tag; the recorded pools for --compare; the last epoch for --compare-runs).

The logic is in `src/pipeline/cross_validate.py` and `src/train/cross_validation.py`.

Usage (from the repository root):
    python scripts/09_cross_validation.py --tag cvbase                          # pool the fold runs of "cvbase"
    python scripts/09_cross_validation.py --compare cvbase cvadapter            # cvadapter − cvbase
    python scripts/09_cross_validation.py --compare cvbase cvbase --mix-weights 0 0.5   # the mix against zero-shot
    python scripts/09_cross_validation.py --compare cvres5lr cvjoint --epoch last      # Pix3D-only against Pix3D + ABO
    python scripts/09_cross_validation.py --compare-runs RUN_A RUN_B                  # two single trainings (run names)
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import cross_validate, extract_features  # noqa: E402
from src.train import cross_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=paths.SPLIT_NAMES, default="unseen_meshes", help="the divided split")
    parser.add_argument("--datasets", nargs="+", choices=extract_features.DATASETS, help="datasets of the trainings (default: data.datasets)")
    parser.add_argument("--backbone", help="backbone of the trainings (default: backbone.name)")
    parser.add_argument("--tag", help="pool the fold runs of this cross-validation")
    parser.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"), help="paired difference TAG_B − TAG_A")
    parser.add_argument("--compare-runs", nargs=2, metavar=("RUN_A", "RUN_B"), help="paired difference RUN_B − RUN_A of two single trainings")
    parser.add_argument("--mix-weights", nargs=2, type=float, metavar=("W_A", "W_B"), help="mix weights of the two sides (default: those of the trainings)")
    parser.add_argument("--epoch", choices=cross_validation.EPOCH_RULES, help="the epoch of the comparison or the pool (see above)")
    parser.add_argument("--mini", action="store_true", help="the runs of mini trainings (test_outputs/train_mini/)")
    add_config_args(parser)
    args = parser.parse_args()
    if sum(bool(x) for x in (args.tag, args.compare, args.compare_runs)) != 1:
        parser.error("give one of --tag, --compare, --compare-runs")
    weights = args.mix_weights or (None, None)
    if args.compare_runs:
        cross_validate.compare_runs(*args.compare_runs, *weights, mini=args.mini, epoch=args.epoch or "last")
        return
    cfg = config_from_args(args)
    datasets = args.datasets or list(cfg["data"]["datasets"])
    backbone = args.backbone or cfg["backbone"]["name"]
    _kind, folds = cross_validate.fold_plan(cfg, args.split, datasets)
    if args.tag:
        cross_validate.pool_and_record(args.split, args.tag, folds, datasets, backbone, args.mini, args.epoch)
    else:
        cross_validate.compare_pools(args.split, *args.compare, folds, backbone, *weights, mini=args.mini, epoch=args.epoch)


if __name__ == "__main__":
    main()
