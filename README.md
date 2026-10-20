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
| Detector and segmenter weights (Grounding DINO, SAM 2.1) | `scripts/00_download_weights.py` | available |
| ABO metadata, mesh geometry, photos, and mesh catalog rows | `scripts/01_download_abo.py` | available |
| Object masks with SAM 2.1 (Pix3D: from the reference box; ABO: from the detector box) | `scripts/02_segment_photos.py --dataset pix3d` | available (ABO cleaning rules not yet reviewed) |
| Query images, catalog, and splits | `scripts/03_make_queries.py --dataset pix3d` | available |
| Renders (25 grey views per mesh) | `scripts/04_render.py --dataset pix3d` (or `abo`) | available |
| Backbone features of the renders and the query images (feature store) | `scripts/05_extract_features.py --dataset pix3d` (or `abo`) | available |
| Training of the embedding heads (one split; best checkpoint by val R@1) | `scripts/06_train.py --split seen_meshes` | available |
| Cross-validation by mesh (one training per fold of the non-test meshes, then the pooled val result) | `scripts/06_train.py --split unseen_meshes --folds all --tag <name>` | available |
| Pool again, or compare two cross-validations (paired, over meshes) | `scripts/09_cross_validation.py --compare <tag a> <tag b>` | available |
| Gallery index of a trained checkpoint (embedded once; checked against the validation R@1) | `scripts/07_build_index.py --run <training run>` | available |
| Evaluation (zero-shot): R@1, R@5, R@10, Cat@1, macro R@1 on the test-free photos or a split part | `scripts/08_evaluate.py --dataset pix3d` | available (the test parts need `--allow-test`) |
| Evaluation (trained): the same metrics with the mixed trained and zero-shot score | `scripts/08_evaluate.py --dataset pix3d --run <training run> --photos val` | available (the test parts need `--allow-test`) |
| Other steps | n/a | to be written |

## Data and model credits

- **Pix3D:** Sun et al., "Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling", CVPR 2018.
- **Amazon Berkeley Objects (ABO):** Collins et al., CVPR 2022. Data © Amazon.com, licensed under CC BY 4.0.
- **DINOv3, planned for v2:** Meta AI, DINOv3 License. The weights are not distributed with this repository.
