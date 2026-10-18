# xdim-cbir: Cross-Dimensional Content-Based Image Retrieval

Find the exact 3D mesh of an object from one photo.

- **Input:** one photo of an object, and a box that marks the object.
- **Output:** a ranked list of 3D meshes from a gallery.

> **Status:** v1 (the baseline) is complete and kept in `legacy/`. v2 is in development; its code goes into `src/`, `scripts/`, and `configs/`.

## Results

| Version | Method | Data | R@1 | R@5 | R@10 | Category accuracy@1 |
|---|---|---|---|---|---|---|
| v1 | ResNet-50 (fine-tuned), 12 icosahedron views, in-batch InfoNCE | Pix3D, 1,510 validation queries | 25.03 % | 56.49 % | 70.33 % | 77.35 % |
| v2 | In development | n/a | n/a | n/a | n/a | n/a |

## Repository layout

```
legacy/      v1, complete and runnable (code unchanged since tag v1)
src/         v2 code (Python package): all functions      [in development]
scripts/     v2 pipeline steps, numbered in order        [in development]
notebooks/   checks and experiments (marimo)             [in development]
configs/     v2 settings (YAML)                          [in development]
runs/        one folder per run: configuration, log, metrics
data/        datasets (not in git; only data/splits/ is tracked)
weights/     pretrained weights and checkpoints (not in git)
envs/        conda environment export
```

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f envs/xdim-env.yml
   conda activate xdim-env
   ```
2. Download Pix3D and extract it to `data/raw/pix3d/`. The folder must contain `pix3d.json`, `img/`, `mask/`, and `model/`. ABO is downloaded by `scripts/01_download_abo.py` (see "Run v2").
3. v2 backbone weights: request access to DINOv3 from Meta and place the weights files in `weights/dinov3/`. The DINOv2 comparison weights go in `weights/dinov2/`. The file names are listed in `configs/backbone.yaml`.
   The model code is not stored in this repository: `torch.hub.load` downloads it from GitHub, at the commit pinned in `configs/backbone.yaml`, on the first use (into `~/.cache/torch/hub/`). Only code is downloaded; the weights always come from the local files.
4. v1 weights: see "Run v1" below.

## Run v1 (baseline)

v1 expects its data and weights in `legacy/data/` and `legacy/weights/`. `legacy/data/pix3d` is a link to `data/raw/pix3d`.

Run the commands from inside `legacy/`, with no `PYTHONPATH` set:

```bash
cd legacy
python src/indexing/evaluate.py
```

## Run v2

Each pipeline step is a Python script in `scripts/`. Run it from the repository root, for example `python scripts/03_make_queries.py --dataset pix3d`. Add `--mini` for a quick check into `test_outputs/`, and `--help` for all options. Each step has a check notebook in `notebooks/` (open it with `marimo edit`), which can also start the mini run with a button.

| Step | Script | Status |
|---|---|---|
| ABO metadata, mesh geometry, and photos | `scripts/01_download_abo.py` | available |
| Query images, catalog, and splits | `scripts/03_make_queries.py --dataset pix3d` | available |
| Renders (25 grey views per mesh) | `scripts/04_render.py --dataset pix3d` | available |
| Other steps | n/a | to be written |

## Data and model credits

- **Pix3D:** Sun et al., "Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling", CVPR 2018.
- **Amazon Berkeley Objects (ABO):** Collins et al., CVPR 2022. Data © Amazon.com, licensed under CC BY 4.0.
- **DINOv3, planned for v2:** Meta AI, DINOv3 License. The weights are not distributed with this repository.
