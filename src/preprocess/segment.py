"""Object masks from photos: a detector finds a box, a segmenter turns the box into a mask.

Two pretrained models do the work (settings in `configs/segment.yaml`, weights in `weights/`):

- Detector, Grounding DINO tiny: photo + text ("sofa.") -> boxes, each with a confidence.
  It is needed only where nobody gives a box: the ABO catalog photos.
- Segmenter, SAM 2.1 (large; `segment.segmenter`): photo + box -> object mask. A box gives SAM the full extent of the
  object, so SAM is less likely to select only a part (a cushion instead of the sofa).
  With a point instead of a box (a click in the demo), the request is ambiguous (the seat? the
  chair? the chair and the floor?), so SAM returns 3 candidate masks and we choose one.

Where the box comes from:

| Use                                   | Box from                          |
|---------------------------------------|-----------------------------------|
| Quality check and end-to-end test     | the reference mask (`box_of_mask`) |
| ABO photos                            | the detector                      |
| Demo                                  | the user                          |

Both models resize the photo themselves (SAM to 1024 x 1024, Grounding DINO to 800 pixels on the
short side) and give results in the pixel coordinates of the original photo. So the result does
not depend on the photo size, and large photos need no extra memory.

Mask format: the same as the Pix3D masks, so that the query step reads both in the same way: one
grey channel (PNG mode "L"), 0 = background (black), 255 = object (white), the size of the photo.
In memory a mask is a bool array (H, W), True = object.

The models load only from the local weight folders, never from the network
(`scripts/00_download_weights.py` downloads them).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from src import paths
from src.preprocess.crop import mask_bbox

# Choice of SAM's mask for a box prompt (`segment.segmenter.box_mask`):
#   "single":  SAM's one-mask output. SAM's authors recommend it for unambiguous prompts (a box).
#   "largest": SAM's three candidate masks; keep the one with the most pixels. This can help when
#              SAM selects a part (the frame of a chair without its cushion).
BOX_MASK_CHOICES = ("single", "largest")

# Point prompt: of SAM's three candidates, keep the largest of those whose predicted quality is at
# most this much below the best. So a clearly worse mask is never chosen only for its size.
POINT_SCORE_MARGIN = 0.05


@dataclass
class MaskResult:
    """One mask from the segmenter."""

    mask: np.ndarray  # bool (H, W), the size of the photo; True = object
    score: float  # SAM's own estimate of the mask quality (its predicted IoU), 0..1


@dataclass
class Detection:
    """One box from the detector."""

    box: tuple[float, float, float, float]  # (x0, y0, x1, y1) in photo pixels
    score: float  # confidence, 0..1
    label: str  # the words of the prompt that the box matches, for example "sofa"


def _as_rgb_array(image: np.ndarray | Image.Image) -> np.ndarray:
    """Return the photo as a uint8 RGB array (H, W, 3)."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 RGB image (H, W, 3), got {image.dtype} {image.shape}")
    # An array made from a PIL image is read-only; torch warns about such arrays, so give it a copy.
    return image if image.flags.writeable else image.copy()


class Segmenter:
    """SAM 2.1: photo + box (or points) -> object mask.

    Example:
        segmenter = Segmenter(cfg["segment"]["segmenter"], device="cuda")
        result = segmenter.mask_from_box(photo, (20, 21, 164, 254))
        result.mask  # bool (H, W)
    """

    def __init__(self, segmenter_cfg: dict, device: str = "cuda"):
        """Load SAM from `segmenter_cfg["folder"]` (relative to the repository root).

        `segmenter_cfg["box_mask"]` selects the mask for a box prompt (see BOX_MASK_CHOICES).
        `segmenter_cfg["precision"]` is "fp16" or "fp32" (default). fp16 applies only on the GPU: the
        weights stay fp32, and PyTorch runs the large matrix products in fp16 ("autocast"), which is
        about 2.5 times faster and gives the same masks. The CPU always computes in fp32.
        """
        # transformers is imported here and not at the top, so that the other functions of this module
        # (mask comparison, mask files) work without loading the large library.
        from transformers import Sam2Model, Sam2Processor

        self.box_mask = segmenter_cfg.get("box_mask", "single")
        if self.box_mask not in BOX_MASK_CHOICES:
            raise ValueError(f"box_mask must be one of {BOX_MASK_CHOICES}, got {self.box_mask!r}")
        folder = paths.resolve(segmenter_cfg["folder"])
        self.device = torch.device(device)
        self.half_precision = segmenter_cfg.get("precision", "fp32") == "fp16" and self.device.type == "cuda"
        self.processor = Sam2Processor.from_pretrained(folder, local_files_only=True)
        # The weight folder describes SAM's video version (config "sam2_video"). The image model uses
        # the same weights; transformers prints a warning about the type, and every weight loads.
        self.model = Sam2Model.from_pretrained(folder, local_files_only=True).to(self.device).eval()

    # SAM works in two stages. The image encoder turns the photo into image features (the slow
    # part, about 90 % of the time). The mask decoder turns the features and a prompt into masks (fast).
    # `_encode` runs the first stage once per photo; `_decode` can then run several times on it.

    @torch.inference_mode()
    def _encode(self, image: np.ndarray, **prompt) -> tuple[dict, list[torch.Tensor]]:
        """Prepare the photo and the prompt, and run the image encoder.

        `prompt` holds the processor's prompt arguments (input_boxes, or input_points and
        input_labels), in photo pixels. Returns the processor output and the image features.
        """
        inputs = self.processor(images=image, return_tensors="pt", **prompt).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half_precision):
            return inputs, self.model.get_image_embeddings(inputs["pixel_values"])

    @torch.inference_mode()
    def _decode(self, inputs: dict, features: list[torch.Tensor], multimask: bool) -> tuple[np.ndarray, np.ndarray]:
        """Run the mask decoder. Return the masks (k, H, W) bool and their predicted quality (k,).

        k is 1 with multimask=False, else 3.
        """
        prompt = {key: inputs[key] for key in ("input_boxes", "input_points", "input_labels") if key in inputs}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half_precision):
            outputs = self.model(image_embeddings=features, multimask_output=multimask, **prompt)
        # SAM predicts masks at 256 x 256; post_process_masks scales them to the photo size and
        # turns them into True/False at the threshold 0 of SAM's mask values (in fp32 on the CPU).
        masks = self.processor.post_process_masks(outputs.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
        return masks[0].numpy().astype(bool), outputs.iou_scores[0, 0].float().cpu().numpy()

    def _choose(self, inputs: dict, features: list[torch.Tensor], choice: str) -> MaskResult:
        """Return the mask of one box mask choice (see BOX_MASK_CHOICES)."""
        if choice == "single":
            masks, scores = self._decode(inputs, features, multimask=False)
            return MaskResult(masks[0], float(scores[0]))
        masks, scores = self._decode(inputs, features, multimask=True)
        best = int(masks.sum(axis=(1, 2)).argmax())  # "largest": the candidate with the most pixels
        return MaskResult(masks[best], float(scores[best]))

    def mask_from_box(
        self, image: np.ndarray | Image.Image, box: tuple[float, float, float, float], choice: str | None = None
    ) -> MaskResult:
        """Return the mask of the object in `box` = (x0, y0, x1, y1), in photo pixels.

        `choice` overrides the mask choice of the settings for this call (see BOX_MASK_CHOICES).
        """
        choice = choice or self.box_mask
        if choice not in BOX_MASK_CHOICES:
            raise ValueError(f"choice must be one of {BOX_MASK_CHOICES}, got {choice!r}")
        inputs, features = self._encode(_as_rgb_array(image), input_boxes=[[[float(v) for v in box]]])  # one photo, one box
        return self._choose(inputs, features, choice)

    def masks_for_each_choice(self, image: np.ndarray | Image.Image, box: tuple[float, float, float, float]) -> dict[str, MaskResult]:
        """Return the mask of every box mask choice, {"single": ..., "largest": ...}.

        The photo is encoded only once, so this is almost as fast as one `mask_from_box` call. For
        experiments that compare the choices.
        """
        inputs, features = self._encode(_as_rgb_array(image), input_boxes=[[[float(v) for v in box]]])
        return {choice: self._choose(inputs, features, choice) for choice in BOX_MASK_CHOICES}

    def mask_from_points(
        self, image: np.ndarray | Image.Image, points: list[tuple[float, float]], labels: list[int] | None = None
    ) -> MaskResult:
        """Return the mask for one or more clicked points (x, y), in photo pixels.

        `labels` has one value per point: 1 = on the object (the default), 0 = not on the object.
        Of the three candidates, the largest one is kept whose predicted quality is within
        POINT_SCORE_MARGIN of the best one.
        """
        image = _as_rgb_array(image)
        labels = labels or [1] * len(points)
        inputs, features = self._encode(
            image,
            input_points=[[[[float(x), float(y)] for x, y in points]]],  # one photo, one object, n points
            input_labels=[[list(labels)]],
        )
        masks, scores = self._decode(inputs, features, multimask=True)
        good = np.flatnonzero(scores >= scores.max() - POINT_SCORE_MARGIN)
        best = int(good[masks[good].sum(axis=(1, 2)).argmax()])
        return MaskResult(masks[best], float(scores[best]))


class Detector:
    """Grounding DINO: photo + text -> boxes with confidences.

    Example:
        detector = Detector(cfg["segment"]["detector"], device="cuda")
        detections = detector.detect(photo, prompt_text("sofa", cfg["segment"]["prompts"]))
        detections[0].box  # the most confident box, or an empty list if nothing was found
    """

    def __init__(self, detector_cfg: dict, device: str = "cuda"):
        """Load Grounding DINO from `detector_cfg["folder"]`; read the two thresholds."""
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        folder = paths.resolve(detector_cfg["folder"])
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = GroundingDinoForObjectDetection.from_pretrained(folder, local_files_only=True).to(self.device).eval()
        self.box_threshold = detector_cfg["box_threshold"]  # minimum confidence of a box
        self.text_threshold = detector_cfg["text_threshold"]  # minimum match of a box to the words

    @torch.inference_mode()
    def detect(self, image: np.ndarray | Image.Image, text: str) -> list[Detection]:
        """Return the boxes that match `text`, most confident first (empty if none passes the thresholds).

        `text` is lower case and ends with a period, for example "sofa." (see `prompt_text`).
        """
        image = _as_rgb_array(image)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        found = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[image.shape[:2]],  # (height, width): boxes in photo pixels
        )[0]
        detections = [
            Detection(tuple(float(v) for v in box), float(score), label)
            for box, score, label in zip(found["boxes"].cpu().tolist(), found["scores"].cpu().tolist(), found["text_labels"])
        ]
        return sorted(detections, key=lambda d: -d.score)


def prompt_text(category: str, prompts: dict[str, str] | None = None) -> str:
    """Return the detector text for a class: the English words, lower case, with a final period.

    `prompts` (`segment.prompts`) gives the words for classes whose folder name is not plain English
    ("tv_stand" -> "tv stand"); other classes use their name with spaces for underscores.
    Grounding DINO expects lower case words, and a period after each object phrase.
    """
    words = (prompts or {}).get(category, category.replace("_", " "))
    return words.lower().rstrip(".") + "."


def box_of_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Return the box (x0, y0, x1, y1) around a mask, for a box prompt (x1 and y1 one past the object)."""
    return tuple(float(v) for v in mask_bbox(mask))


def edges_touched(box: tuple[float, float, float, float], width: int, height: int, tolerance: float = 2.0) -> int:
    """Count the photo edges (0 to 4) that a box touches, within `tolerance` pixels.

    A close-up photo of a detail usually fills the frame, so its box touches 3 or 4 edges.
    """
    x0, y0, x1, y1 = box
    return int(x0 <= tolerance) + int(y0 <= tolerance) + int(x1 >= width - tolerance) + int(y1 >= height - tolerance)


# ---------------------------------------------------------------------------------------------
# Mask comparison (for the quality check against reference masks)
# ---------------------------------------------------------------------------------------------


def mask_iou(mask: np.ndarray, reference: np.ndarray, ignore: np.ndarray | None = None) -> float:
    """Return the IoU of two masks: (pixels in both) / (pixels in either); 1.0 = identical.

    Pixels where `ignore` is True do not count. Two empty masks give 1.0.
    """
    if ignore is not None:
        mask, reference = mask & ~ignore, reference & ~ignore
    union = np.count_nonzero(mask | reference)
    return 1.0 if union == 0 else np.count_nonzero(mask & reference) / union


def edge_band(mask: np.ndarray, width: int) -> np.ndarray:
    """Return the pixels within `width` pixels of the mask's edge, inside or outside (bool (H, W)).

    The quality check ignores this band for an "edge-tolerant" IoU: a reference mask that is a few
    pixels off the photo (or a slightly different edge) then does not count as an error. Without the
    band, a shift of 2 pixels on a chair leg 5 pixels wide already halves the IoU of that leg.
    "Within `width` pixels" uses a square neighbourhood (also diagonally `width` pixels). The band is
    the mask grown by `width` minus the mask shrunk by `width`; both use a running maximum (or minimum)
    along the rows and then the columns, so the cost does not depend on `width`.
    """
    if width <= 0:
        return np.zeros_like(mask, dtype=bool)
    size = 2 * width + 1
    values = mask.astype(np.uint8)
    grown = ndimage.maximum_filter1d(ndimage.maximum_filter1d(values, size, axis=0), size, axis=1)
    # Pixels outside the photo count as object here, so that an object cut by the photo border is
    # not shrunk from that border (the border is not an edge of the object).
    shrunk = ndimage.minimum_filter1d(ndimage.minimum_filter1d(values, size, axis=0, mode="nearest"), size, axis=1, mode="nearest")
    return grown.astype(bool) & ~shrunk.astype(bool)


def edge_band_width(mask: np.ndarray, fraction: float) -> int:
    """Return the band width in pixels for a photo: `fraction` of its longer side, at least 1."""
    return max(1, round(fraction * max(mask.shape)))


def grow_mask(mask: np.ndarray, pixels: int, box: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Return the mask grown by `pixels` in every direction (a square neighbourhood), optionally kept inside `box`.

    A grown mask keeps a thin ring of real background around the object. If the mask misses a thin
    part (a leg, the edge of a cushion), the grown mask often covers it again. Keeping the result inside
    the box stops the growth from adding background beyond the object's extent.
    The growth uses a running maximum along the rows and then along the columns, so its cost does not
    depend on `pixels`.
    """
    if pixels <= 0:
        grown = mask.copy()
    else:
        size = 2 * pixels + 1
        grown = ndimage.maximum_filter1d(mask.astype(np.uint8), size, axis=0)
        grown = ndimage.maximum_filter1d(grown, size, axis=1).astype(bool)
    if box is not None:
        grown &= box_mask(mask.shape, box)
    return grown


def box_mask(shape: tuple[int, int], box: tuple[float, float, float, float]) -> np.ndarray:
    """Return a mask that is True inside `box` = (x0, y0, x1, y1) and False outside, of size `shape` = (H, W).

    Used as the "mask" of a query that keeps all pixels of the box (the background inside the box too).
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    inside = np.zeros(shape, dtype=bool)
    inside[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = True
    return inside


# ---------------------------------------------------------------------------------------------
# Mask files
# ---------------------------------------------------------------------------------------------


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a bool mask as a PNG in the Pix3D format (mode "L", 0 = background, 255 = object).

    The file is written under a temporary name and then renamed, so that an interrupted run never
    leaves a half-written mask.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(partial, format="PNG")  # 2-D uint8 -> mode "L"
    partial.replace(path)
    return path


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask file (any grey or colour PNG) as bool (H, W); True where the value is above 127."""
    with Image.open(paths.resolve(path)) as image:
        return np.asarray(image.convert("L")) > 127
