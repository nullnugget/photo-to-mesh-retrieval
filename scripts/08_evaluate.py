"""Evaluate retrieval on a group of photos of one dataset, and record the results as a run.

Zero-shot mode: the stored backbone features of the query photos are compared with the stored
features of the views of every gallery mesh, without training. One call evaluates every combination
of the given query sets, feature recipes, and scorings, on the same photos.

Photo groups (--photos):
    test_free   photos that are in no test part of any split. The lighting, backbone, recipe,
                segmenter, and scoring are shared by all splits, so they are selected only on these
                photos (a selection fits its photos a little). No --split.
    train, val  a part of --split.
    test        the test part of --split: the reported number, evaluated once after all
                selections are fixed. Needs --allow-test.

Outputs: runs/<run_name>/ (config, metrics.json, one CSV per setting with the rank of every photo)
and one row per setting in runs/registry.csv. A mini run (--mini) uses the mini feature files of
`scripts/05_extract_features.py --mini` and writes into test_outputs/evaluate_mini/.

The logic is in `src/pipeline/run_evaluation.py` and `src/retrieval/evaluate.py`.

Usage (from the repository root):
    python scripts/08_evaluate.py --dataset pix3d                                   # test-free photos, both query sets, cls, view score
    python scripts/08_evaluate.py --dataset pix3d --recipes cls masked_mean cls_masked_mean --scorings view mesh combined
    python scripts/08_evaluate.py --dataset abo --photos val --split unseen_meshes
    python scripts/08_evaluate.py --dataset pix3d --photos test --split seen_meshes --allow-test
    python scripts/08_evaluate.py --dataset pix3d --mini
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.models.backbones import FEATURE_RECIPES  # noqa: E402
from src.pipeline import extract_features, run_evaluation  # noqa: E402
from src.retrieval.evaluate import SCORINGS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=extract_features.DATASETS, default="pix3d", help="dataset to evaluate")
    parser.add_argument("--photos", choices=run_evaluation.PHOTO_GROUPS, default=run_evaluation.TEST_FREE, help="photo group (default: test_free)")
    parser.add_argument("--split", choices=paths.SPLIT_NAMES, help="split of the photo group (not for test_free)")
    parser.add_argument("--query-sets", nargs="+", choices=["queries_reference", "queries_sam"], help="query sets (default: all of the dataset)")
    parser.add_argument("--recipes", nargs="+", choices=FEATURE_RECIPES, help="feature recipes (default: backbone.feature_recipe)")
    parser.add_argument("--scorings", nargs="+", choices=SCORINGS, help="scorings (default: view)")
    parser.add_argument("--alpha", type=float, help="weight of the view score in the combined score (default: train.scoring.alpha)")
    parser.add_argument("--backbone", help="backbone of the feature files (default: backbone.name)")
    parser.add_argument("--allow-test", action="store_true", help="allow the test part (the final evaluation, once)")
    parser.add_argument("--tag", help="extra word for the run name")
    parser.add_argument("--mini", action="store_true", help="the mini feature files; outputs into test_outputs/")
    add_config_args(parser)
    args = parser.parse_args()
    run_evaluation.run(args.dataset, config_from_args(args), photos=args.photos, split=args.split,
                       query_sets=args.query_sets, recipes=args.recipes, scorings=args.scorings, alpha=args.alpha,
                       backbone=args.backbone, allow_test=args.allow_test, tag=args.tag, mini=args.mini)


if __name__ == "__main__":
    main()
