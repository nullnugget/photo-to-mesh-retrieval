"""Render 25 grey views of every gallery mesh of one dataset.

The meshes come from the catalog (`data/derived/catalog/meshes.csv`); the settings (rig, light
setup, image size, material) from `configs/render.yaml`. Each mesh gets the folder of its catalog
row, for example `data/derived/renders/pix3d/chair/IKEA_POANG_2/view_00.png` ... `view_24.png`.

The step can be stopped and started again: meshes whose folder exists are skipped. The settings are
saved in `data/derived/renders/render_config.yaml`; with changed settings the step stops, unless
--overwrite is given (then every mesh is rendered again).

A mini run (--mini) renders 2 meshes per category into `test_outputs/render_mini/`. The check
notebook `notebooks/02_render_check.py` can start a mini run with a button and shows the results.

The logic is in `src/pipeline/render_meshes.py`.

Usage (from the repository root):
    python scripts/04_render.py --dataset pix3d            # all Pix3D meshes, about 5 minutes
    python scripts/04_render.py --dataset pix3d --mini     # quick check
    python scripts/04_render.py --dataset pix3d --overwrite
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import render_meshes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=["pix3d", "abo"], default="pix3d", help="dataset to render")
    parser.add_argument("--mini", action="store_true", help="2 meshes per category, into test_outputs/")
    parser.add_argument("--overwrite", action="store_true", help="render every mesh again (also with changed settings)")
    add_config_args(parser)
    args = parser.parse_args()
    render_meshes.run(args.dataset, config_from_args(args), mini=args.mini, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
