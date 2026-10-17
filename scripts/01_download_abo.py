"""Download the part of the ABO dataset that the project uses.

Stages (each skips the files that exist already, so a stopped run continues where it stopped):

1. metadata: the 18 ABO metadata files (about 90 MB) into `data/raw/abo/metadata/`.
2. selection: apply the class rules of `configs/data.yaml` (27 classes, 4,943 meshes, 29,263
   photos) and write the manifest into `data/raw/abo/manifest/`. This stage always runs.
3. meshes: only the geometry of each mesh (about 5 % of each `.glb` file), saved as
   `data/raw/abo/meshes/<class>/<model id>.ply`, with its glTF description `<model id>.gltf.json`.
4. photos: the catalog photos at full resolution, `data/raw/abo/images/<class>/<image id>.jpg`.

The full download is about 21 GB and takes about 40 minutes. Each stage shows a progress bar
with the remaining time. Ctrl-C is safe: run the same command again to continue.

A mini run (--mini) takes the first 3 meshes of each class and only their photos, and writes them
to `test_outputs/download_abo_mini/`. The check notebook `notebooks/03_abo_check.py` can start a
mini run with a button and then shows the results.

The logic is in `src/pipeline/download_abo.py`.

Usage (from the repository root):
    python scripts/01_download_abo.py                     # everything
    python scripts/01_download_abo.py --mini              # quick check, about a minute
    python scripts/01_download_abo.py --stages photos     # one stage only
"""

import argparse
import sys
from pathlib import Path

# Make the project package "src" importable when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_config_args, config_from_args  # noqa: E402
from src.pipeline import download_abo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mini", action="store_true", help="3 meshes per class and their photos, into test_outputs/")
    parser.add_argument("--stages", nargs="+", choices=download_abo.STAGES, default=list(download_abo.STAGES), help="stages to run")
    parser.add_argument("--workers", type=int, default=None, help="parallel downloads (default: data.abo.download_workers)")
    add_config_args(parser)
    args = parser.parse_args()
    download_abo.run(config_from_args(args), mini=args.mini, stages=tuple(args.stages), workers=args.workers)


if __name__ == "__main__":
    main()
