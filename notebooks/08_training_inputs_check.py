import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 08: Training Inputs Check

    This notebook shows what the embedding heads will see during training: the query images of the
    training photos after the random changes (`src/data/augment.py`, `src/data/datasets.py`).

    Each training photo has one or more prepared query images (224 × 224, grey background, tight
    crop). For Pix3D, these are the image cut out with the dataset's own mask ("reference") and the
    image cut out with the SAM mask ("sam"); at each use, one of them is picked at random. ABO photos
    have only the SAM image. Then the random changes follow: mask noise (the outline moves inwards,
    small holes), a tight crop again, colour jitter on the object only, sometimes grey, a small
    rotation, a zoom out, and sometimes an occluding patch.

    It shows:

    1. the training photos of the chosen split and their images;
    2. a sheet: for a few random photos, the reference and SAM images, then several changed versions;
       the sheet is also saved to `test_outputs/training_inputs_check/`.

    The speed of the data loader is not measured here: the training step logs it in every epoch
    (images per second and the share of the time that the GPU waits for the data loader).

    **How to run:** from the repository root, `marimo edit notebooks/08_training_inputs_check.py`, or
    `python notebooks/08_training_inputs_check.py -- --dataset abo` (as a script; without `--dataset`
    it checks Pix3D). It reads only the prepared query images; it needs no GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports the libraries and the project modules and sets the sample sizes. The cell
    after it chooses the dataset; as a script, the dataset comes from `--dataset`.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw

    # The notebook is in <repository>/notebooks/, so the repository root is one level up.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes the project package "src" importable
    from src import paths  # noqa: E402
    from src.config import load_config  # noqa: E402
    from src.data.augment import QueryAugment  # noqa: E402
    from src.data.datasets import training_photos  # noqa: E402

    CFG = load_config()
    SPLIT, PART = "seen_meshes", "train"
    SHEET_PHOTOS = 6  # photos on the sheet
    SHEET_CHANGES = 6  # changed versions of each photo
    SEED = 0
    OUTPUT_DIR = paths.TEST_OUTPUTS_DIR / "training_inputs_check"
    return CFG, Image, ImageDraw, OUTPUT_DIR, PART, QueryAugment, SEED, SHEET_CHANGES, SHEET_PHOTOS, SPLIT, np, training_photos


@app.cell
def _(mo):
    # In the editor, choose here. As a script: python notebooks/08_training_inputs_check.py -- --dataset abo
    dataset_choice = mo.ui.dropdown(["pix3d", "abo"], value=mo.cli_args().get("dataset") or "pix3d", label="dataset to check")
    dataset_choice
    return (dataset_choice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The training photos and their images

    The next cell lists the training photos of the split part (`training_photos`) and counts their images.

    | Field | Meaning | Expected |
    |:---|:---|:---|
    | `photos` | Training photos with at least one image | all photos of the part |
    | `images` | Images found, by dataset and kind | Pix3D: reference and sam for every photo; ABO: sam |
    | `missing_images` | Images that do not exist | none |
    | `left_out` | Photos without any image | 0 |
    """)
    return


@app.cell
def _(CFG, PART, SPLIT, dataset_choice, mo, training_photos):
    PHOTOS, _report = training_photos([dataset_choice.value], SPLIT, PART, CFG["train"]["query_images"])
    _summary = {"split": SPLIT, "part": PART, "photos": _report["photos"], "images": _report["images"],
                "missing_images": _report["missing_images"], "left_out": len(_report["left_out"])}
    print("PHOTOS", _summary)
    mo.ui.table([{k: str(v) for k, v in _summary.items()}], selection=None)
    return (PHOTOS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Sample sheet

    One row per random photo. The first columns are its prepared images without changes (reference,
    then SAM for Pix3D); the next columns are changed versions, each from a new random draw (the
    image kind is also drawn each time). Look for: the object stays recognisable; the background stays
    a flat grey; the outline sometimes moves inwards or has a small hole; colours vary only on the
    object; some versions are grey, a little tilted, smaller, or have a coloured patch. The sheet is
    saved as `test_outputs/training_inputs_check/<dataset>_sheet.png`.
    """)
    return


@app.cell
def _(CFG, Image, ImageDraw, OUTPUT_DIR, PHOTOS, QueryAugment, SEED, SHEET_CHANGES, SHEET_PHOTOS, dataset_choice, mo, np):
    _augment = QueryAugment(CFG["train"]["augment"], CFG["data"]["background_rgb"], CFG["data"]["image_size"])
    _rng = np.random.default_rng(SEED)
    _tile = 160
    _columns = 2 + SHEET_CHANGES
    _sheet = Image.new("RGB", (_columns * _tile, SHEET_PHOTOS * (_tile + 16)), "white")
    _draw = ImageDraw.Draw(_sheet)
    for _row, _index in enumerate(_rng.choice(len(PHOTOS), size=SHEET_PHOTOS, replace=False)):
        _photo = PHOTOS[_index]
        _originals = [np.array(Image.open(f).convert("RGB")) for f in _photo.images]
        _tiles = [(f"{_k} (no change)", _img) for _k, _img in zip(("first image", "second image"), _originals)]
        _tiles += [(f"change {_c + 1}", _augment(_originals[int(_rng.integers(len(_originals)))], _rng)) for _c in range(SHEET_CHANGES)]
        for _col, (_label, _img) in enumerate(_tiles[:_columns]):
            _sheet.paste(Image.fromarray(_img).resize((_tile, _tile)), (_col * _tile, _row * (_tile + 16) + 16))
        _draw.text((4, _row * (_tile + 16) + 2), f"{_photo.query_id}  ({len(_photo.images)} image(s): first column(s) unchanged)", fill="black")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_FILE = OUTPUT_DIR / f"{dataset_choice.value}_sheet.png"
    _sheet.save(SHEET_FILE)
    print("SHEET", str(SHEET_FILE))
    mo.image(_sheet)
    return


if __name__ == "__main__":
    app.run()
