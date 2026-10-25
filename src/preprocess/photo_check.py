"""Photo check: does a product photo show the whole object, or only a part of it (or no photo of it at all)?

Some ABO extra images (the images after a listing's main image) show a detail: a leg, a drawer pull, the
fabric, a corner; others are a line drawing with dimensions, an infographic, or a swatch. Such a photo
cannot be matched to the whole mesh, and as a training photo it teaches a wrong match. The box and mask
signals of the segment step cannot find them: the detector is confident on a close-up (a sofa arm is a
part of a sofa), and the crop fills the frame like a whole object.

A text-image model answers the question directly. SigLIP base (settings `segment.photo_check`, weights in
`weights/`) gives every (image, text) pair a score; the texts describe a whole object or a bad photo of
the photo's class:

| Kind  | Texts (class "sofa")                                                              |
|-------|-----------------------------------------------------------------------------------|
| whole | a photo of a sofa; a product photo of a whole sofa; a photo of a sofa in a room;  |
|       | a product photo of a sofa with its measurements                                   |
| bad   | a close-up photo of a part of a sofa; a close-up photo of the fabric of a sofa;   |
|       | a black and white line drawing of a sofa; a photo of a detail of a sofa           |

Many product photos show the whole object with dimension arrows and numbers; they are good photos. A
first text "a line drawing of a sofa with measurements" among the bad texts took 54 such photos for bad
(of 55 good photos it removed in the review sample), so the measurements belong to a whole text.

The bad score of a photo is the share of the bad texts in the softmax over all eight scores (0 to 1).
SigLIP never takes part in the search: it only decides which photos are used, so the check does not
favour the photos that the search finds easy (DINOv3 would).

The model loads only from the local weight folder (`scripts/00_download_weights.py --models photo_check`).
The tokenizer is read from `tokenizer.json` (the fast tokenizer), so the `sentencepiece` package is not
needed; the texts are lower case without punctuation, as SigLIP's own tokenizer would make them.
"""

import numpy as np
import torch
from PIL import Image

from src import paths

WHOLE_TEXTS = ("a photo of a {}", "a product photo of a whole {}", "a photo of a {} in a room",
               "a product photo of a {} with its measurements")
BAD_TEXTS = ("a close-up photo of a part of a {}", "a close-up photo of the fabric of a {}",
             "a black and white line drawing of a {}", "a photo of a detail of a {}")
TEXT_LENGTH = 64  # SigLIP base was trained with texts padded to 64 tokens


def class_words(category: str, prompts: dict[str, str] | None = None) -> str:
    """The English words of a class ("tv_stand" -> "tv stand"; `segment.prompts` for the others)."""
    return (prompts or {}).get(category, category.replace("_", " ")).lower()


class PhotoCheck:
    """SigLIP base, ready to give the bad score of product photos."""

    def __init__(self, check_cfg: dict, prompts: dict[str, str] | None = None, device: str = "cuda"):
        from transformers import PreTrainedTokenizerFast, SiglipImageProcessor, SiglipModel

        folder = paths.resolve(check_cfg["folder"])
        self.device = device
        self.prompts = prompts
        self.model = SiglipModel.from_pretrained(folder).to(device).eval()
        self.processor = SiglipImageProcessor.from_pretrained(folder)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(folder / "tokenizer.json"),
                                                 pad_token="</s>", eos_token="</s>", unk_token="<unk>")
        self._texts: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _text_embeddings(self, category: str) -> torch.Tensor:
        """The unit text embeddings of the whole texts, then the bad texts, of one class (cached)."""
        if category not in self._texts:
            words = class_words(category, self.prompts)
            texts = [t.format(words) for t in WHOLE_TEXTS + BAD_TEXTS]
            tokens = self.tokenizer(texts, padding="max_length", max_length=TEXT_LENGTH, return_tensors="pt").to(self.device)
            embedded = self.model.get_text_features(input_ids=tokens["input_ids"])
            self._texts[category] = embedded / embedded.norm(dim=-1, keepdim=True)
        return self._texts[category]

    @torch.no_grad()
    def bad_scores(self, images: list[Image.Image], categories: list[str]) -> np.ndarray:
        """The bad score (0 to 1) of each image, against the texts of its class."""
        pixels = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"].to(self.device)
        embedded = self.model.get_image_features(pixel_values=pixels)
        embedded = embedded / embedded.norm(dim=-1, keepdim=True)
        scale, bias = self.model.logit_scale.exp(), self.model.logit_bias
        scores = []
        for vector, category in zip(embedded, categories):
            logits = vector @ self._text_embeddings(category).T * scale + bias
            scores.append(float(logits.softmax(-1)[len(WHOLE_TEXTS):].sum()))
        return np.array(scores)
