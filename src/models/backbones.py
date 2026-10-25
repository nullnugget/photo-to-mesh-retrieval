"""Frozen pretrained backbones (DINOv3, DINOv2) that turn images into feature vectors.

A backbone is a large vision network trained by Meta on very many images. The project does not
train it ("frozen"); it only reads its output. For a 224 x 224 image, a ViT backbone with 16-pixel
patches returns:

- one class token: a vector that summarizes the whole image;
- 14 x 14 = 196 patch tokens: one vector per 16 x 16 square of the image;
- (4 extra tokens, not used).

Feature recipes (`backbone.feature_recipe` in `configs/backbone.yaml`) make one vector per image:

- "cls": the class token.
- "masked_mean": the average of the patch tokens on the object. Each patch counts with the share
  of its pixels that belong to the object, so the grey background does not count.
- "cls_masked_mean": both, each scaled to length 1, joined into one vector of twice the length.

Every feature vector is scaled to length 1, so that the dot product of two features is their
cosine similarity (1 = same direction, 0 = unrelated).

The two basic parts ("cls" and "masked_mean") come from the same pass through the backbone.
`FeatureExtractor.extract_parts` returns both, and `combine_recipe` makes any recipe from them. The
feature store keeps the two parts, so one extraction serves every recipe.

Loading: `torch.hub.load` takes the model code from GitHub at a fixed commit (downloaded once into
`~/.cache/torch/hub/`), builds the network with `pretrained=False` (no download of weights), and
the weights come from the local file named in the configuration.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src import paths

FEATURE_RECIPES = ("cls", "masked_mean", "cls_masked_mean")
FEATURE_PARTS = ("cls", "masked_mean")  # the basic parts; every recipe is made from them


@dataclass(frozen=True)
class BackboneSpec:
    """One entry of `backbone.available`."""

    name: str  # builder name in the model repository, for example "dinov3_vitb16"
    hub: str  # model code for torch.hub: "<github owner>/<repo>:<commit>"
    weights: str  # local weights file, relative to the repository root
    dim: int  # length of the class token and of each patch token
    patch_size: int  # side length of one patch in pixels


def backbone_spec(backbone_cfg: dict, name: str | None = None) -> BackboneSpec:
    """Return the spec of the backbone `name`, or of the active one (`backbone.name`)."""
    name = name or backbone_cfg["name"]
    entry = backbone_cfg["available"][name]
    return BackboneSpec(name, entry["hub"], entry["weights"], entry["dim"], entry["patch_size"])


def load_backbone(spec: BackboneSpec, device: str | torch.device = "cuda") -> torch.nn.Module:
    """Build the backbone with torch.hub and load its local weights. Returns the model in evaluation mode."""
    # pretrained=False: the builder downloads no weights. trust_repo=True: no interactive question.
    model = torch.hub.load(spec.hub, spec.name, pretrained=False, trust_repo=True)
    state_dict = torch.load(paths.resolve(spec.weights), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)  # strict: every weight must match exactly
    return model.eval().to(device)


def object_masks(images: np.ndarray, background_rgb: tuple[int, int, int]) -> np.ndarray:
    """Return True where a pixel differs from the background grey: bool (N, H, W).

    Queries and renders have an exactly uniform grey background, so the object is every pixel that
    is not exactly this grey. (An object pixel of exactly this grey counts as background; this is
    rare and changes the patch weights very little.)
    """
    return (images != np.asarray(background_rgb, dtype=np.uint8)).any(axis=-1)


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector (the last axis) to length 1, in float32."""
    vectors = vectors.astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=1e-12)


def combine_recipe(parts: dict[str, np.ndarray], recipe: str) -> np.ndarray:
    """Make the feature vectors of one recipe from the basic parts: float32, unit length.

    Args:
        parts: "cls" and/or "masked_mean" -> vectors (..., dim). The leading axes can be anything,
            for example (N,) for queries or (meshes, views) for renders. The vectors can be float16
            (as in the feature store); each part is scaled to length 1 again in float32.
        recipe: One of FEATURE_RECIPES.

    Returns:
        (..., dim) for "cls" and "masked_mean"; (..., 2 x dim) for "cls_masked_mean".
    """
    if recipe not in FEATURE_RECIPES:
        raise ValueError(f"recipe must be one of {FEATURE_RECIPES}, got {recipe!r}")
    if recipe != "cls_masked_mean":
        return _unit(parts[recipe])
    # Two unit vectors joined have length sqrt(2); scaling to length 1 gives both parts equal weight.
    return _unit(np.concatenate([_unit(parts["cls"]), _unit(parts["masked_mean"])], axis=-1))


class FeatureExtractor:
    """Turns batches of images into feature vectors with one backbone and one recipe."""

    def __init__(self, backbone_cfg: dict, name: str | None = None, recipe: str | None = None, device: str = "cuda"):
        """
        Args:
            backbone_cfg: The `backbone` section of the configuration.
            name: Backbone to use (None = `backbone.name`).
            recipe: Feature recipe (None = `backbone.feature_recipe`).
            device: "cuda" or "cpu".
        """
        self.spec = backbone_spec(backbone_cfg, name)
        self.recipe = recipe or backbone_cfg["feature_recipe"]
        if self.recipe not in FEATURE_RECIPES:
            raise ValueError(f"recipe must be one of {FEATURE_RECIPES}, got {self.recipe!r}")
        self.device = torch.device(device)
        self.model = load_backbone(self.spec, self.device)
        self.image_size = backbone_cfg["image_size"]
        self.batch_size = backbone_cfg["batch_size"]
        self.half_precision = backbone_cfg["precision"] == "fp16" and self.device.type == "cuda"
        self.mean = torch.tensor(backbone_cfg["normalize"]["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(backbone_cfg["normalize"]["std"], device=self.device).view(1, 3, 1, 1)

    @property
    def dim(self) -> int:
        """Length of the feature vectors that this extractor returns."""
        return self.spec.dim * (2 if self.recipe == "cls_masked_mean" else 1)

    def _prepare(self, images: np.ndarray) -> torch.Tensor:
        """uint8 (n, H, W, 3) -> normalized float tensor (n, 3, size, size) on the device."""
        batch = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        if batch.shape[-1] != self.image_size or batch.shape[-2] != self.image_size:
            batch = F.interpolate(batch, size=(self.image_size, self.image_size), mode="bilinear", antialias=True)
        return (batch - self.mean) / self.std  # the colour normalization the backbone was trained with

    def _patch_weights(self, masks: np.ndarray) -> torch.Tensor:
        """Share of object pixels in every patch: (n, patches), in the order of the patch tokens."""
        mask = torch.from_numpy(masks).to(self.device).float()[:, None]  # (n, 1, H, W)
        mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="nearest")
        grid = self.image_size // self.spec.patch_size  # patches per side, for example 14
        shares = F.adaptive_avg_pool2d(mask, grid)  # (n, 1, grid, grid): mean of the mask in each patch
        return shares.flatten(1)  # row by row, the same order as the patch tokens

    @torch.inference_mode()
    def extract_parts(self, images: np.ndarray, masks: np.ndarray | None, parts=FEATURE_PARTS) -> dict[str, np.ndarray]:
        """Return the basic parts of every image, each scaled to length 1: {part: float32 (N, spec.dim)}.

        One pass through the backbone gives all the parts, so asking for both costs no extra time.

        Args:
            images: uint8 (N, H, W, 3), RGB.
            masks: bool (N, H, W), True on the object; usually `object_masks(images, background_rgb)`.
                Needed for "masked_mean"; can be None for "cls" only.
            parts: The parts to return, from FEATURE_PARTS.
        """
        if "masked_mean" in parts and masks is None:
            raise ValueError("the part 'masked_mean' needs object masks")
        outputs: dict[str, list] = {part: [] for part in parts}
        for start in range(0, len(images), self.batch_size):
            batch = self._prepare(images[start : start + self.batch_size])
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half_precision):
                output = self.model.forward_features(batch)
            if "cls" in parts:
                outputs["cls"].append(F.normalize(output["x_norm_clstoken"].float(), dim=-1).cpu())
            if "masked_mean" in parts:
                weights = self._patch_weights(masks[start : start + self.batch_size])  # (n, patches)
                weights = torch.where(weights.sum(1, keepdim=True) > 0, weights, torch.ones_like(weights))  # no object: plain mean
                tokens = output["x_norm_patchtokens"].float()  # (n, patches, dim)
                mean = (weights[..., None] * tokens).sum(1) / weights.sum(1, keepdim=True)
                outputs["masked_mean"].append(F.normalize(mean, dim=-1).cpu())
        return {part: torch.cat(vectors).numpy().astype(np.float32) for part, vectors in outputs.items()}

    @torch.inference_mode()
    def extract_patches(self, images: np.ndarray, masks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the patch tokens and the object share of every patch, on the device (for the patch store).

        Args:
            images: uint8 (N, H, W, 3), RGB; at most `batch_size` images (one pass).
            masks: bool (N, H, W), True on the object.

        Returns:
            tokens: float32 (N, patches, spec.dim), the normalized patch tokens (not scaled to length 1).
            shares: float32 (N, patches), the share of object pixels of each patch, in token order.
        """
        if len(images) > self.batch_size:
            raise ValueError(f"at most {self.batch_size} images per call, got {len(images)}")
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half_precision):
            output = self.model.forward_features(self._prepare(images))
        return output["x_norm_patchtokens"].float(), self._patch_weights(masks)

    def extract(self, images: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
        """Return one unit-length feature vector per image, with this extractor's recipe: float32 (N, dim).

        Args:
            images: uint8 (N, H, W, 3), RGB.
            masks: bool (N, H, W), True on the object. Needed for the masked recipes; usually
                `object_masks(images, background_rgb)`.
        """
        if self.recipe != "cls" and masks is None:
            raise ValueError(f"recipe {self.recipe!r} needs object masks")
        parts = ("cls",) if self.recipe == "cls" else ("masked_mean",) if self.recipe == "masked_mean" else FEATURE_PARTS
        return combine_recipe(self.extract_parts(images, masks, parts), self.recipe)
