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
    # 00: Environment Check

    This notebook checks that the pretrained backbones run on this computer, with the installed
    versions of Python, PyTorch, and CUDA. A backbone is the large, frozen network that turns an
    image into a feature vector. The project uses DINOv3 and compares it with DINOv2.

    For each backbone, the notebook records:

    - whether the model loads, and its size;
    - the shapes of its outputs;
    - whether the same input always gives the same output;
    - whether half precision (fp16, bf16) gives the same result as full precision (fp32);
    - its speed and its peak GPU memory for a batch of 64 images;
    - a small sanity check: does one photo of a chair look more similar to renders of its own
      chair than to renders of another chair?

    **How to run**

    - As a script, from the repository root: `python notebooks/00_env_check.py`
    - Interactively: `marimo edit notebooks/00_env_check.py`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and settings

    The next cell imports every library that the notebook uses and defines the settings:

    - the folders of the weights (`weights/`) and of the test images;
    - the ImageNet colour normalization, which all these backbones expect;
    - the list of backbones to check. It comes from `configs/backbone.yaml`, so that the pinned code
      versions and the weights files are defined in one place. Each entry gives the name of the
      builder function, the source of its code for `torch.hub.load`, and its weights file.
    """)
    return


@app.cell
def _():
    import json
    import platform
    import sys
    import time
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    import torchvision
    import torchvision.io as tvio
    import torchvision.transforms.functional as TF

    # Folders. The notebook is in <repository>/notebooks/, so the repository root is one level up.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))  # makes the project package "src" importable
    from src.config import load_config  # noqa: E402 (needs the path line above)

    LEGACY_DATA = PROJECT_ROOT / "legacy" / "data"  # v1 images, used as test inputs (see section 4)
    PIX3D_JSON = PROJECT_ROOT / "data" / "raw" / "pix3d" / "pix3d.json"  # Pix3D annotations

    # ImageNet colour normalization: subtract this mean and divide by this standard deviation, per channel.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Use the GPU when one is available.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Backbones to check, from configs/backbone.yaml:
    # (name of the builder function, torch.hub source "<owner>/<repo>:<commit>", weights file).
    BACKBONES = [
        (name, entry["hub"], PROJECT_ROOT / entry["weights"])
        for name, entry in load_config(["backbone.yaml"])["backbone"]["available"].items()
    ]
    return (
        BACKBONES,
        DEVICE,
        F,
        IMAGENET_MEAN,
        IMAGENET_STD,
        LEGACY_DATA,
        PIX3D_JSON,
        TF,
        json,
        platform,
        sys,
        time,
        torch,
        torchvision,
        tvio,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Environment

    The next cell shows the software versions and the GPU. The backbones need CUDA (the GPU) for the
    speed test. "bf16 supported" tells whether the GPU can compute in the bfloat16 number format.
    """)
    return


@app.cell
def _(DEVICE, mo, platform, sys, torch, torchvision):
    _has_gpu = torch.cuda.is_available()
    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch CUDA build": torch.version.cuda,  # CUDA version that PyTorch was built for
        "CUDA available": _has_gpu,
        "GPU": torch.cuda.get_device_name(0) if _has_gpu else "-",
        "GPU memory (GiB)": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if _has_gpu else 0,
        "bf16 supported": torch.cuda.is_bf16_supported() if _has_gpu else False,
        "torchvision": torchvision.__version__,
        "device": str(DEVICE),
    }
    print("ENV", env_info)  # printed, so that a script run also shows it
    mo.md("\n".join(f"- **{key}:** {value}" for key, value in env_info.items()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Helper functions

    The next cell defines four functions.

    **`load_backbone`** builds a backbone with `torch.hub.load` and loads its local weights.

    - `torch.hub.load("<owner>/<repo>:<commit>", name)` downloads the model **code** of that exact
      commit from GitHub, once, into `~/.cache/torch/hub/`, and reuses it after that. The download
      contains no weights. It reads the repository file `hubconf.py`, which lists the models that
      the repository offers, and calls the builder function `name`.
    - DINOv3's `hubconf.py` also imports Meta's evaluation tools, which need the package
      `torchmetrics`. This package is installed in the environment. Importing that code does not
      create any model and does not download any weights.
    - The builder gets `pretrained=False`, so it does not download weights. The notebook then reads
      the local weights file with `torch.load`. Thus the only weights on disk are the files in
      `weights/`.

    **`load_image`** reads an image, resizes it to 224 × 224 pixels, and applies the ImageNet
    normalization.

    **`tokens`** runs a backbone on a batch of images and returns its three kinds of output tokens:

    - the **class token**: one vector that summarizes the whole image;
    - the **patch tokens**: one vector for each image patch (a 16 × 16 or 14 × 14 pixel square);
    - the **extra tokens** (called "storage" tokens in DINOv3 and "register" tokens in DINOv2):
      a few learned vectors that the model uses as working memory. The project does not use them.

    **`check_backbone`** runs all the checks for one backbone and returns one row of results
    (section 5 explains each field).
    """)
    return


@app.cell
def _(
    DEVICE,
    F,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TF,
    time,
    torch,
    tvio,
):
    def load_backbone(name, hub_source, weights_file):
        """Build the backbone `name` with torch.hub, and load the local weights in `weights_file`.

        Args:
            name: Name of the builder function, for example "dinov3_vitb16".
            hub_source: Source of the code, "<github owner>/<repo>:<commit>".
            weights_file: Local weights file.
        """
        # Build the network with random weights. The code comes from the torch hub cache (downloaded
        # on the first call). pretrained=False: the builder downloads no weights.
        # trust_repo=True: accept this repository without an interactive question.
        model = torch.hub.load(hub_source, name, pretrained=False, trust_repo=True)
        # Load the pretrained weights from the local file.
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)  # strict: every weight must match exactly
        return model.eval()  # evaluation mode: no dropout or other training behaviour

    def load_image(path, size=224):
        """Read an image as RGB, resize it to size x size, and apply the ImageNet normalization."""
        image = tvio.read_image(str(path), mode=tvio.ImageReadMode.RGB).float() / 255.0  # values 0..1
        image = TF.resize(image, [size, size], antialias=True)
        return TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

    def tokens(model, images, dtype=None):
        """Run the backbone on a batch of images.

        Args:
            model: A loaded backbone.
            images: Tensor of shape (batch, 3, 224, 224).
            dtype: None for full precision (fp32), or torch.float16 / torch.bfloat16 for half precision.

        Returns:
            (class token, patch tokens, extra tokens), all as fp32 tensors.
            Shapes: (batch, D), (batch, patches, D), (batch, 4, D). D is the feature length.
        """
        with torch.inference_mode():  # no gradients: faster, less memory
            if dtype is None:
                out = model.forward_features(images)
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model.forward_features(images)
        class_token = out["x_norm_clstoken"].float()
        patch_tokens = out["x_norm_patchtokens"].float()
        # DINOv3 calls the extra tokens "storage tokens"; DINOv2 calls them "register tokens".
        extra = out.get("x_storage_tokens", out.get("x_norm_regtokens"))
        return class_token, patch_tokens, (extra.float() if extra is not None else None)

    def check_backbone(name, hub_source, weights_file, query, own_views, other_views):
        """Run all checks for one backbone. Return one dict of results (see section 5)."""
        result = {"backbone": name}
        try:
            # Start each backbone with an empty GPU, so that its peak memory is measured alone.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 1. Load the model and count its parameters.
            start = time.perf_counter()
            model = load_backbone(name, hub_source, weights_file).to(DEVICE)
            result["load_seconds"] = round(time.perf_counter() - start, 2)
            result["params_millions"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

            # 2. Output shapes, in full precision (fp32).
            query = query.to(DEVICE)
            class32, patches32, extra32 = tokens(model, query)
            result["class_token_shape"] = tuple(class32.shape)
            result["patch_tokens_shape"] = tuple(patches32.shape)
            result["extra_tokens_shape"] = tuple(extra32.shape) if extra32 is not None else None

            # 3. Repeatability: run the same image again and compare.
            class32_again, _, _ = tokens(model, query)
            result["repeat_max_abs_diff"] = float((class32 - class32_again).abs().max())

            # 4. Half precision against full precision. A cosine similarity of 1.0 means the same direction.
            class16, _, _ = tokens(model, query, torch.float16)
            class_bf16, _, _ = tokens(model, query, torch.bfloat16)
            result["cos_fp16_vs_fp32"] = round(float(F.cosine_similarity(class16, class32)), 5)
            result["cos_bf16_vs_fp32"] = round(float(F.cosine_similarity(class_bf16, class32)), 5)
            result["fp16_finite"] = bool(torch.isfinite(class16).all())  # no overflow (inf) or NaN

            # 5. Sanity check: the query against the renders of its own shape and of another chair.
            own, _, _ = tokens(model, own_views.to(DEVICE), torch.float16)
            other, _, _ = tokens(model, other_views.to(DEVICE), torch.float16)
            result["sim_own_shape"] = round(float(F.cosine_similarity(class16, own).max()), 3)
            result["sim_other_shape"] = round(float(F.cosine_similarity(class16, other).max()), 3)

            # 6. Speed and peak memory: a batch of 64 images in fp16.
            batch = query.repeat(64, 1, 1, 1)
            for _ in range(2):  # warm-up runs: the first runs include one-time GPU setup
                tokens(model, batch, torch.float16)
            torch.cuda.synchronize()  # wait for the GPU, so that the timer starts at the right moment
            start = time.perf_counter()
            for _ in range(5):
                tokens(model, batch, torch.float16)
            torch.cuda.synchronize()
            result["images_per_second"] = round(5 * 64 / (time.perf_counter() - start), 1)
            result["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

            result["status"] = "OK"
            del model, batch
        except Exception as error:  # record the failure and continue with the next backbone
            result["status"] = f"FAILED: {type(error).__name__}: {error}"
        torch.cuda.empty_cache()
        return result

    return check_backbone, load_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Test inputs

    The next cell loads one Pix3D photo of a chair (`img/chair/0001.png`), the 12 renders of its
    own 3D shape, and the 12 renders of another chair.

    These are v1 images from `legacy/data/`: the photo is the v1 preprocessed version, and the renders
    come from the v1 renderer. They are temporary stand-ins. When the v2 queries and renders exist,
    they replace these images, and the notebook no longer depends on v1.
    """)
    return


@app.cell
def _(LEGACY_DATA, PIX3D_JSON, json, load_image, torch):
    # Find the 3D shape of the test photo in the Pix3D annotations.
    with open(PIX3D_JSON) as _f:
        _annotations = json.load(_f)
    _photo = next(entry for entry in _annotations if entry["img"] == "img/chair/0001.png")
    own_shape = _photo["model"].split("/")[2]  # "model/chair/IKEA_POANG_2/model.obj" -> "IKEA_POANG_2"

    # Choose another chair that also has v1 renders (the first in alphabetical order).
    _chairs = sorted({entry["model"].split("/")[2] for entry in _annotations if entry["category"] == "chair"})
    other_shape = next(c for c in _chairs if c != own_shape and (LEGACY_DATA / "renders" / "chair" / c).exists())

    # Load the images. Shapes: query (1, 3, 224, 224); views (12, 3, 224, 224).
    _renders = LEGACY_DATA / "renders" / "chair"
    query = load_image(LEGACY_DATA / "processed" / "img" / "chair" / "0001.png").unsqueeze(0)
    own_views = torch.stack([load_image(_renders / own_shape / f"view_{v:02d}.png") for v in range(12)])
    other_views = torch.stack([load_image(_renders / other_shape / f"view_{v:02d}.png") for v in range(12)])
    print("INPUTS", {"query": "img/chair/0001.png", "own_shape": own_shape, "other_shape": other_shape})
    return other_shape, other_views, own_shape, own_views, query


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Checks for each backbone

    The next cell runs `check_backbone` for every backbone and collects one row of results for each.
    The fields of a row:

    | Field | Meaning | Good value |
    |:---|:---|:---|
    | `backbone` | Name of the backbone | — |
    | `load_seconds` | Time to build the model and load its weights | a few seconds |
    | `params_millions` | Number of trainable values in the model, in millions (M = million) | as expected for the model size |
    | `class_token_shape` | Shape of the class token: (images, D), where D is the feature length. The class token is one extra vector that the backbone adds to the image patches. Through the attention layers it collects information from all patches, so at the output it summarizes the whole image. It is the usual image feature for retrieval: images with similar content give class tokens that point in similar directions | (1, 384), (1, 768), or (1, 1024) |
    | `patch_tokens_shape` | Shape of the patch tokens: (images, patches, D). The backbone cuts the image into small squares (patches) and gives one vector to each. At the output, each patch token describes its square in the context of the whole image (for example "the edge of a chair leg"). The patch tokens keep the layout of the image, so they can describe only the object: the average of the tokens inside the object mask ignores the background. A 224-pixel image gives 14 × 14 = 196 patches of 16 pixels (DINOv3), or 16 × 16 = 256 patches of 14 pixels (DINOv2) | (1, 196, D) or (1, 256, D) |
    | `extra_tokens_shape` | Shape of the extra tokens: (images, 4, D). DINOv3 calls them "storage" tokens and DINOv2 "register" tokens. They are 4 learned vectors with no position in the image. The backbone uses them as working memory for global information, which keeps the patch tokens clean (without them, some background patches carry this information and show as noise). The project does not use them as features | (1, 4, D) |
    | `repeat_max_abs_diff` | Largest difference between two runs on the same image | 0.0 (exactly repeatable) |
    | `cos_fp16_vs_fp32` | Cosine similarity between the class token in fp16 and in fp32 | 1.0 (no loss from half precision) |
    | `cos_bf16_vs_fp32` | The same for bf16 | close to 1.0 |
    | `fp16_finite` | True if the fp16 output has no infinite or undefined (NaN) values | True |
    | `sim_own_shape` | Highest cosine similarity between the photo and the 12 renders of its own chair | higher than `sim_other_shape` |
    | `sim_other_shape` | Highest cosine similarity between the photo and the 12 renders of another chair | lower than `sim_own_shape` |
    | `images_per_second` | Speed for batches of 64 images in fp16 | — |
    | `peak_vram_gib` | Highest GPU memory in use during the checks, in GiB | well below the GPU size |
    | `status` | "OK", or "FAILED" with the error message | OK |

    The sanity check uses one photo only. It shows that the backbone works, not how good it is.
    """)
    return


@app.cell
def _(BACKBONES, check_backbone, other_views, own_views, query):
    results = []
    for _name, _hub_source, _weights_file in BACKBONES:
        _row = check_backbone(_name, _hub_source, _weights_file, query, own_views, other_views)
        print("RESULT", _row)  # printed, so that a script run also shows each row
        results.append(_row)
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Summary

    The next cell shows whether all backbones passed, and the full results table.
    """)
    return


@app.cell
def _(mo, other_shape, own_shape, results):
    _all_ok = all(row["status"] == "OK" for row in results)
    _summary = (
        f"**Result: {'all backbones OK' if _all_ok else 'at least one backbone FAILED'}**\n\n"
        f"Test inputs: photo `img/chair/0001.png`, own chair `{own_shape}`, other chair `{other_shape}`."
    )
    print("SUMMARY", "ALL_OK" if _all_ok else "FAILURES")
    mo.vstack([mo.md(_summary), mo.ui.table(results, selection=None)])
    return


if __name__ == "__main__":
    app.run()
