"""Random changes to a training query image, made again at every use of the photo.

A training photo appears in every epoch. If it looked the same each time, the heads could learn
the exact pixels of the photo instead of the mesh. Random changes that keep the mesh recognisable
make every use a little different. Each change imitates a real difference between photos and
renders, or between the training photos and the photos of real use:

| Step | Change                              | Why                                                          |
|------|-------------------------------------|--------------------------------------------------------------|
| 1    | mask noise (erosion, small holes)   | SAM outlines are a few pixels off (`src/preprocess/masks.py`) |
| 2    | tight crop again                    | the queries and renders are always cropped tight to the object |
| 3    | colour jitter, object pixels only   | photos have colours and lighting that the grey renders do not |
| 4    | grey (with probability p)           | makes the heads match on form, not on colour                  |
| 5    | rotation (± degrees, grey fill)     | real photos are a little tilted                               |
| 6    | zoom out (object 85 to 100 % of the frame) | small framing differences                              |
| 7    | occluding patch (with probability p) | other objects can hide a part of the object                  |
| 8    | horizontal flip (with probability p) | off by default: some meshes are not mirror-symmetric         |

The background stays exactly the background grey after every step (the colour changes touch only
the object pixels; rotation and zoom fill with grey), because the renders have exactly this grey
and a tinted background would be a cue that the gallery never shows.

The input and output are uint8 RGB arrays (H, W, 3). The random values come from a NumPy
generator that the caller owns (one per data loader worker), so that a run can be repeated.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.models.backbones import object_masks
from src.preprocess.crop import EmptyMaskError, crop_to_square
from src.preprocess.masks import mask_noise


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0  # (3, H, W), 0..1


def _to_array(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).numpy()


class QueryAugment:
    """The random changes of one training query image (see the module description)."""

    def __init__(self, augment_cfg: dict, background_rgb, image_size: int):
        """
        Args:
            augment_cfg: `train.augment` of the configuration.
            background_rgb: The background grey, for example (128, 128, 128).
            image_size: Side of the output image, for example 224.
        """
        self.cfg = augment_cfg
        self.background = np.asarray(background_rgb, dtype=np.uint8)
        self.fill = [float(v) / 255.0 for v in self.background]  # the grey as 0..1 values, for TF fills
        self.size = image_size

    def _object(self, image: np.ndarray) -> np.ndarray:
        return object_masks(image[None], tuple(self.background))[0]  # True where the pixel is not the grey

    def _colour(self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Colour jitter and grey on the object pixels; the background stays exactly grey."""
        jitter = self.cfg["color_jitter"]
        tensor = _to_tensor(image)
        tensor = TF.adjust_brightness(tensor, rng.uniform(1 - jitter["brightness"], 1 + jitter["brightness"]))
        tensor = TF.adjust_contrast(tensor, rng.uniform(1 - jitter["contrast"], 1 + jitter["contrast"]))
        tensor = TF.adjust_saturation(tensor, rng.uniform(1 - jitter["saturation"], 1 + jitter["saturation"]))
        tensor = TF.adjust_hue(tensor, rng.uniform(-jitter["hue"], jitter["hue"]))
        if rng.random() < self.cfg["grayscale_p"]:
            tensor = TF.rgb_to_grayscale(tensor, num_output_channels=3)
        return np.where(mask[..., None], _to_array(tensor), self.background)

    def _geometry(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Rotation and zoom out, with grey fill.

        The rotation turns the image and its object mask together, and every pixel that is less than
        half object afterwards is set to the exact grey. Without this step the background is not
        exactly grey: torchvision blends the image edge with black before it adds the fill, which
        leaves faint dark lines along the edges of the turned frame.
        """
        tensor = _to_tensor(image)
        angle = rng.uniform(-self.cfg["rotation_deg"], self.cfg["rotation_deg"])
        weight = torch.from_numpy(self._object(image)).float()[None]  # (1, H, W): 1 = object
        tensor = TF.rotate(tensor, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=self.fill)
        weight = TF.rotate(weight, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=[0.0])
        tensor = torch.where(weight >= 0.5, tensor, torch.tensor(self.fill).view(3, 1, 1))
        scale = rng.uniform(self.cfg["zoom_out_min"], 1.0)
        inner = max(1, int(round(self.size * scale)))
        if inner < self.size:  # shrink the image and place it at a random spot on a grey square
            small = TF.resize(tensor, [inner, inner], antialias=True)
            top, left = (int(rng.integers(0, self.size - inner + 1)) for _ in range(2))
            canvas = torch.tensor(self.fill).view(3, 1, 1).expand(3, self.size, self.size).clone()
            canvas[:, top : top + inner, left : left + inner] = small
            tensor = canvas
        return _to_array(tensor)

    def _occlude(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """With probability erasing_p, cover a random rectangle of the object with one random colour.

        The rectangle covers 2 to 10 % of the image; only its object pixels change, so the patch
        looks like something in front of the object, and the background stays grey.
        """
        if rng.random() >= self.cfg["erasing_p"]:
            return image
        mask = self._object(image)
        area = rng.uniform(0.02, 0.10) * self.size * self.size
        ratio = np.exp(rng.uniform(np.log(0.3), np.log(3.3)))  # width / height, spread evenly on a log scale
        height = int(min(self.size, round(np.sqrt(area / ratio))))
        width = int(min(self.size, round(np.sqrt(area * ratio))))
        top, left = int(rng.integers(0, self.size - height + 1)), int(rng.integers(0, self.size - width + 1))
        patch = np.zeros_like(mask)
        patch[top : top + height, left : left + width] = True
        result = image.copy()
        result[patch & mask] = rng.integers(0, 256, size=3, dtype=np.uint8)
        return result

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a randomly changed copy of a query image: uint8 (size, size, 3)."""
        mask = self._object(image)
        if self.cfg["mask_noise"]["enabled"]:  # 1 and 2: noise on the outline, then the tight crop again
            noisy = mask_noise(mask, rng, self.cfg["mask_noise"])
            if noisy is not mask:
                try:
                    image = crop_to_square(image, noisy, tuple(self.background), 0.0, self.size)
                except EmptyMaskError:  # cannot happen (mask_noise never returns an empty mask); keep the image
                    pass
                mask = self._object(image)
        image = self._colour(image, mask, rng)  # 3 and 4
        image = self._geometry(image, rng)  # 5 and 6
        image = self._occlude(image, rng)  # 7
        if rng.random() < self.cfg["hflip_p"]:  # 8
            image = np.ascontiguousarray(image[:, ::-1])
        return image
