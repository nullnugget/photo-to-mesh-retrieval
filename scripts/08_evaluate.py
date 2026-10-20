"""Evaluate retrieval on a group of photos of one dataset, and record the results as a run.

Zero-shot mode: the stored backbone features of the query photos are compared with the stored
features of the views of every gallery mesh, without training. One call evaluates every combination
of the given query sets, feature recipes, and scorings, on the same photos.

Trained mode (--run <training run>): the checkpoint of the run and its gallery index (build it
first with scripts/07_build_index.py). The final score is mix_weight × trained score + (1 −
mix_weight) × zero-shot score; the mix weight, the alpha, and the zero-shot scoring are those of
the training (--mix-weights evaluates other mix weights too: 1 = trained alone, 0 = zero-shot).
The photos are of the training split; the test-free photos do not apply.

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
    python scripts/08_evaluate.py --dataset pix3d                                   # test-free photos, both query sets, the configured recipe and scoring
    python scripts/08_evaluate.py --dataset pix3d --recipes cls masked_mean cls_masked_mean --scorings view mesh combined
    python scripts/08_evaluate.py --dataset abo --photos val --split unseen_meshes
    python scripts/08_evaluate.py --dataset pix3d --photos test --split seen_meshes --allow-test
    python scripts/08_evaluate.py --dataset pix3d --mini
    python scripts/08_evaluate.py --dataset pix3d --run 2026-10-20_pix3d_dinov3_vitb16_seen_meshes_mix05 --photos val --mix-weights 0.5 1 0
    python scripts/08_evaluate.py --dataset pix3d --run 2026-10-20_pix3d_dinov3_vitb16_seen_meshes_mix05 --photos test --allow-test
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths  # noqa: E402
from src.config import add_config_args, config_from_args  # noqa: E402
from src.models.backbones import FEATURE_RECIPES  # noqa: E402
from src.pipeline import build_index, extract_features, run_evaluation  # noqa: E402
from src.retrieval.evaluate import SCORINGS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=extract_features.DATASETS, default="pix3d", help="dataset to evaluate")
    parser.add_argument("--photos", choices=run_evaluation.PHOTO_GROUPS, default=run_evaluation.TEST_FREE, help="photo group (default: test_free)")
    parser.add_argument("--split", choices=paths.SPLIT_NAMES, help="split of the photo group (not for test_free)")
    parser.add_argument("--query-sets", nargs="+", choices=["queries_reference", "queries_sam"], help="query sets (default: all of the dataset)")
    parser.add_argument("--recipes", nargs="+", choices=FEATURE_RECIPES, help="feature recipes (default: backbone.feature_recipe)")
    parser.add_argument("--scorings", nargs="+", choices=SCORINGS, help="scorings (default: backbone.zero_shot.scoring)")
    parser.add_argument("--alpha", type=float, help="weight of the view score in the combined score (default: backbone.zero_shot.alpha)")
    parser.add_argument("--backbone", help="backbone of the feature files (default: backbone.name)")
    parser.add_argument("--allow-test", action="store_true", help="allow the test part (the final evaluation, once)")
    parser.add_argument("--tag", help="extra word for the run name")
    parser.add_argument("--mini", action="store_true", help="the mini feature files (trained mode: a mini training); outputs into test_outputs/")
    trained = parser.add_argument_group("trained mode")
    trained.add_argument("--run", help="training run name: evaluate its checkpoint with its gallery index")
    trained.add_argument("--which", choices=build_index.CHECKPOINT_KINDS, default="best", help="checkpoint of the run")
    trained.add_argument("--mix-weights", nargs="+", type=float, help="mix weights (default: that of the training)")
    add_config_args(parser)
    args = parser.parse_args()
    if args.run:
        unused = [name for name in ("recipes", "scorings", "alpha", "backbone") if getattr(args, name) is not None]
        if unused:
            parser.error(f"--{', --'.join(unused)}: not for trained mode (the checkpoint fixes these settings)")
        run_evaluation.run_trained(args.dataset, config_from_args(args), args.run, which=args.which,
                                   photos=args.photos if args.photos != run_evaluation.TEST_FREE else "val", split=args.split,
                                   query_sets=args.query_sets, mix_weights=args.mix_weights, allow_test=args.allow_test,
                                   tag=args.tag, mini=args.mini)
        return
    if args.mix_weights or args.which != "best":
        parser.error("--mix-weights and --which need --run")
    run_evaluation.run(args.dataset, config_from_args(args), photos=args.photos, split=args.split,
                       query_sets=args.query_sets, recipes=args.recipes, scorings=args.scorings, alpha=args.alpha,
                       backbone=args.backbone, allow_test=args.allow_test, tag=args.tag, mini=args.mini)


if __name__ == "__main__":
    main()
