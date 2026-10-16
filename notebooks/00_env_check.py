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
    # 00: Environment check (task T1.1, experiment E1)

    This notebook checks that the frozen backbones for v2 run in the current environment.

    For each backbone it records:
    - load success and parameter count;
    - output token shapes;
    - repeatability;
    - fp16 and bf16 accuracy against fp32;
    - speed and peak VRAM for a batch of 64 images.

    A small sanity check compares one Pix3D query with the v1 renders of its own shape and of another chair.

    Run it as a script (`python notebooks/00_env_check.py`), or open it with `marimo edit notebooks/00_env_check.py`.
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

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    THIRD_PARTY = PROJECT_ROOT / "third_party"
    WEIGHTS = PROJECT_ROOT / "weights"
    LEGACY_DATA = PROJECT_ROOT / "legacy" / "data"
    PIX3D_JSON = PROJECT_ROOT / "data" / "raw" / "pix3d" / "pix3d.json"
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Backbones to check: (hub entry name, local code repository, local weights file)
    BACKBONES = [
        ("dinov3_vits16", THIRD_PARTY / "dinov3", WEIGHTS / "dinov3" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
        ("dinov3_vitb16", THIRD_PARTY / "dinov3", WEIGHTS / "dinov3" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        ("dinov3_vitl16", THIRD_PARTY / "dinov3", WEIGHTS / "dinov3" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
        ("dinov2_vitb14_reg", THIRD_PARTY / "dinov2", WEIGHTS / "dinov2" / "dinov2_vitb14_reg4_pretrain.pth"),
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


@app.cell
def _(DEVICE, mo, platform, sys, torch, torchvision):
    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch CUDA build": torch.version.cuda,
        "CUDA available": torch.cuda.is_available(),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
        "GPU memory (GiB)": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else 0,
        "bf16 supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "torchvision": torchvision.__version__,
        "device": str(DEVICE),
    }
    print("ENV", env_info)
    mo.md("## Environment\n\n" + "\n".join(f"- **{k}:** {v}" for k, v in env_info.items()))
    return (env_info,)


@app.cell
def _(IMAGENET_MEAN, IMAGENET_STD, TF, sys, torch, tvio):
    import importlib

    def load_backbone(name, repo_dir, weights_file):
        """Build the network from the local code, then load the local weights.

        The builder is imported from `<repo>/<package>/hub/backbones.py` directly, not through `hubconf.py`.
        `hubconf.py` of DINOv3 also imports the evaluation heads, which need `torchmetrics`.
        `pretrained=False` and `torch.load` prevent a copy of the weights into the torch hub cache.
        """
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        builder = getattr(importlib.import_module(f"{repo_dir.name}.hub.backbones"), name)
        model = builder(pretrained=False)
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model.eval()

    def load_image(path, size=224):
        """Read an image as RGB, resize to size x size, and apply the ImageNet normalization."""
        img = tvio.read_image(str(path), mode=tvio.ImageReadMode.RGB).float() / 255.0
        img = TF.resize(img, [size, size], antialias=True)
        return TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)

    def tokens(model, x, dtype=None):
        """Run the backbone. Return the class token, patch tokens, and register/storage tokens (or None)."""
        with torch.inference_mode():
            if dtype is None:
                out = model.forward_features(x)
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model.forward_features(x)
        cls = out["x_norm_clstoken"].float()
        patches = out["x_norm_patchtokens"].float()
        extra = out.get("x_storage_tokens", out.get("x_norm_regtokens"))
        return cls, patches, (extra.float() if extra is not None else None)

    return load_backbone, load_image, tokens


@app.cell
def _(LEGACY_DATA, PIX3D_JSON, json, load_image, torch):
    # Test inputs: one Pix3D query (v1 preprocessed), the 12 v1 renders of its shape, and 12 renders of another chair.
    with open(PIX3D_JSON) as _f:
        _meta = json.load(_f)
    _item = next(m for m in _meta if m["img"] == "img/chair/0001.png")
    gt_shape = _item["model"].split("/")[2]
    _chairs = sorted({m["model"].split("/")[2] for m in _meta if m["category"] == "chair"})
    other_shape = next(c for c in _chairs if c != gt_shape and (LEGACY_DATA / "renders" / "chair" / c).exists())

    query = load_image(LEGACY_DATA / "processed" / "img" / "chair" / "0001.png").unsqueeze(0)
    gt_views = torch.stack([load_image(LEGACY_DATA / "renders" / "chair" / gt_shape / f"view_{v:02d}.png") for v in range(12)])
    other_views = torch.stack([load_image(LEGACY_DATA / "renders" / "chair" / other_shape / f"view_{v:02d}.png") for v in range(12)])
    print("INPUTS", {"query": "img/chair/0001.png", "gt_shape": gt_shape, "other_shape": other_shape})
    return gt_shape, gt_views, other_shape, other_views, query


@app.cell
def _(
    BACKBONES,
    DEVICE,
    F,
    gt_views,
    load_backbone,
    other_views,
    query,
    time,
    tokens,
    torch,
):
    results = []
    for _name, _repo, _weights in BACKBONES:
        _r = {"backbone": _name}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _t = time.perf_counter()
            _model = load_backbone(_name, _repo, _weights).to(DEVICE)
            _r["load_s"] = round(time.perf_counter() - _t, 2)
            _r["params_M"] = round(sum(p.numel() for p in _model.parameters()) / 1e6, 1)

            _q = query.to(DEVICE)
            _cls32, _patch32, _extra32 = tokens(_model, _q)
            _r["cls_shape"] = tuple(_cls32.shape)
            _r["patch_shape"] = tuple(_patch32.shape)
            _r["extra_tokens"] = tuple(_extra32.shape) if _extra32 is not None else None

            # Repeatability: the same input twice gives the same output
            _cls32b, _, _ = tokens(_model, _q)
            _r["repeat_max_abs_diff"] = float((_cls32 - _cls32b).abs().max())

            # Reduced precision against fp32 (cosine similarity of the class token)
            _cls16, _, _ = tokens(_model, _q, torch.float16)
            _clsbf, _, _ = tokens(_model, _q, torch.bfloat16)
            _r["cos_fp16_vs_fp32"] = round(float(F.cosine_similarity(_cls16, _cls32)), 5)
            _r["cos_bf16_vs_fp32"] = round(float(F.cosine_similarity(_clsbf, _cls32)), 5)
            _r["fp16_finite"] = bool(torch.isfinite(_cls16).all())

            # Sanity: query against the v1 renders of its own shape and of another chair (fp16)
            _gt, _, _ = tokens(_model, gt_views.to(DEVICE), torch.float16)
            _ot, _, _ = tokens(_model, other_views.to(DEVICE), torch.float16)
            _sg = F.cosine_similarity(_cls16, _gt)
            _so = F.cosine_similarity(_cls16, _ot)
            _r["sim_own_shape_max"] = round(float(_sg.max()), 3)
            _r["sim_other_shape_max"] = round(float(_so.max()), 3)

            # Speed and peak VRAM: batch of 64 images at 224 x 224, fp16, no gradients
            _batch = query.repeat(64, 1, 1, 1).to(DEVICE)
            for _ in range(2):
                tokens(_model, _batch, torch.float16)
            torch.cuda.synchronize()
            _t = time.perf_counter()
            for _ in range(5):
                tokens(_model, _batch, torch.float16)
            torch.cuda.synchronize()
            _r["imgs_per_s_fp16_b64"] = round(5 * 64 / (time.perf_counter() - _t), 1)
            _r["peak_vram_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            _r["status"] = "OK"
            del _model, _batch
        except Exception as _e:
            _r["status"] = f"FAILED: {type(_e).__name__}: {_e}"
        torch.cuda.empty_cache()
        print("RESULT", _r)
        results.append(_r)
    return (results,)


@app.cell
def _(gt_shape, mo, other_shape, results):
    _ok = all(r["status"] == "OK" for r in results)
    _summary = (
        f"## Result: {'ALL BACKBONES OK' if _ok else 'AT LEAST ONE BACKBONE FAILED'}\n\n"
        f"Sanity inputs: query `img/chair/0001.png`, own shape `{gt_shape}`, other chair `{other_shape}`."
    )
    print("SUMMARY", "ALL_OK" if _ok else "FAILURES")
    mo.vstack([mo.md(_summary), mo.ui.table(results, selection=None)])
    return


if __name__ == "__main__":
    app.run()
