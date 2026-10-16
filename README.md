# xdim-cbir: Cross-Dimensional Content-Based Image Retrieval

Find the exact 3D shape of an object from one photo.

- **Input:** one photo of an object, and a box that marks the object.
- **Output:** a ranked list of 3D shapes from a gallery of meshes.

> **Status:** v1 (the baseline) is complete and kept in `legacy/`. v2 is in development; its code goes into `src/`, `scripts/`, and `configs/`.

## Results

| Version | Method | Data | R@1 | R@5 | R@10 | Category accuracy@1 |
|---|---|---|---|---|---|---|
| v1 | ResNet-50 (fine-tuned), 12 icosahedron views, in-batch InfoNCE | Pix3D, 1,510 validation queries | 25.03 % | 56.49 % | 70.33 % | 77.35 % |
| v2 | In development | n/a | n/a | n/a | n/a | n/a |

## Repository layout

```
legacy/      v1, complete and runnable (code unchanged since tag v1)
src/         v2 code (Python package)                    [in development]
scripts/     v2 commands, numbered in pipeline order     [in development]
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
2. Download Pix3D and extract it to `data/raw/pix3d/`. The folder must contain `pix3d.json`, `img/`, `mask/`, and `model/`.
3. Weights: see the setup of each version below.

## Run v1 (baseline)

v1 expects its data and weights in `legacy/data/` and `legacy/weights/`. `legacy/data/pix3d` is a link to `data/raw/pix3d`.

Run the commands from inside `legacy/`, with no `PYTHONPATH` set:

```bash
cd legacy
python src/indexing/evaluate.py
```

## Run v2

To be written.

## Data and model credits

- **Pix3D:** Sun et al., "Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling", CVPR 2018.
- **Amazon Berkeley Objects (ABO), planned for v2:** Collins et al., CVPR 2022. Data © Amazon.com, licensed under CC BY 4.0.
- **DINOv3, planned for v2:** Meta AI, DINOv3 License. The weights are not distributed with this repository.
