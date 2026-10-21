"""The training loop of the embedding heads.

One training step (batch of B photos; the training gallery has M meshes of V views):

    1. B changed query images (`src/data/datasets.py`)        uint8 (B, 224, 224, 3), CPU workers
    2. DINOv3, frozen, no gradients                            query features (B, 768)
    3. query projector                                          q (B, 256)
    4. view projector on ALL views of ALL training meshes      e (M, V, 256)   from the stored features
    5. view aggregator with random view dropout                g (M, 256)
    6. combined scores                                          s (B, M)
    7. full-gallery InfoNCE + lambda × category loss           one number
    8. backward and AdamW: only the three heads change

The training gallery holds only the meshes of the training photos: a mesh whose photos are all in
val or test is never seen in training, not even as a wrong answer (as in real use, where a new
mesh is added after the training). The validation gallery holds every mesh of the dataset.

Validation after each epoch: the stored features of the val photos (no changes, both mask sets for
Pix3D) and of all views are embedded with the current heads, and R@1, R@5, R@10, and Cat@1 are
calculated against the full gallery of each dataset. The selection value is the mean R@1 over the
val query sets; the epoch with the highest value gives the best checkpoint.

The final score mixes the trained score with the zero-shot score of the raw features:

    final = mix_weight × trained score + (1 − mix_weight) × zero-shot score

Both come from the same backbone pass of the photo: the raw feature is compared with the stored raw
view features (zero-shot), and its query embedding with the trained view and mesh embeddings. The
heads learn mostly the meshes that they train on; the zero-shot part keeps the general knowledge of
the frozen backbone, which matters for meshes that the training never saw. mix_weight 1.0 is the
trained score alone. The validation reports the mixed R@1 (which selects the epoch) and the R@1 of
the trained score alone ("trained_R@1").

Learning rate: a linear rise from 0 during the warm-up epochs, then a cosine fall to 0 at the end.
Weight decay applies to the weight matrices only, not to biases or to the rating vector.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import paths
from src.data.splits import load_split
from src.models.backbones import FeatureExtractor, object_masks
from src.models.embedding_heads import EmbeddingHeads, random_view_mask, scores
from src.models.feature_store import read_features
from src.pipeline.extract_features import QUERY_SETS
from src.retrieval import metrics
from src.retrieval.evaluate import score_meshes
from src.train.losses import gallery_info_nce, gallery_targets


@dataclass
class Gallery:
    """Stored view features of a set of meshes, on the training device."""

    views: torch.Tensor  # float16 (M, V, feature_dim), unit length per view (the chosen recipe)
    mesh_ids: list[str]
    categories: torch.Tensor  # int64 (M,), category numbers


@dataclass
class ValSet:
    """The stored features of the val photos of one dataset and one query set."""

    name: str  # for example "pix3d_queries_reference"
    dataset: str
    features: torch.Tensor  # float16 (Q, feature_dim)
    targets: torch.Tensor  # int64 (Q,), column of the correct mesh in the dataset's gallery
    categories: torch.Tensor  # int64 (Q,)
    zero_shot_scores: torch.Tensor  # float32 (Q, M): zero-shot score of every gallery mesh (raw features; fixed)
    zero_shot_r1: float  # R@1 of the raw features (zero-shot setting), for comparison
    query_ids: np.ndarray | None = None  # (Q,): for the per-photo record (cross-validation)
    mesh_ids: np.ndarray | None = None  # (Q,): the correct mesh of each photo
    category_names: np.ndarray | None = None  # (Q,)


def load_galleries(datasets: list[str], backbone: str, recipe: str, category_index: dict[str, int],
                   features_root, device) -> tuple[dict[str, Gallery], list]:
    """Return the full gallery of each dataset (every mesh of its renders file) and the files read."""
    galleries, files = {}, []
    for dataset in datasets:
        file = paths.features_file(backbone, dataset, "renders", features_root)
        table = read_features(file)
        files.append(file)
        galleries[dataset] = Gallery(
            views=torch.from_numpy(table.vectors(recipe)).to(device, torch.float16),
            mesh_ids=table.ids.tolist(),
            categories=torch.tensor([category_index[c] for c in table.labels["category"]], device=device),
        )
    return galleries, files


def training_gallery(galleries: dict[str, Gallery], mesh_ids: set[str]) -> Gallery:
    """Return the meshes of the training photos, from the full galleries of all datasets."""
    keep = [(dataset, i) for dataset, gallery in galleries.items() for i, m in enumerate(gallery.mesh_ids) if m in mesh_ids]
    return Gallery(
        views=torch.stack([galleries[d].views[i] for d, i in keep]),
        mesh_ids=[galleries[d].mesh_ids[i] for d, i in keep],
        categories=torch.stack([galleries[d].categories[i] for d, i in keep]),
    )


def load_val_sets(datasets: list[str], split: str, backbone: str, recipe: str, galleries: dict[str, Gallery],
                  category_index: dict[str, int], zero_shot_cfg: dict, features_root, device) -> list[ValSet]:
    """Return the val photos of every query set of every dataset, as stored features.

    A query set without val photos is left out (in a joint fold by category, ABO has no photos of
    the Pix3D category "misc").
    """
    sets = []
    for dataset in datasets:
        wanted = load_split(paths.split_file(dataset, split))["parts"]["val"]
        gallery = galleries[dataset]
        for query_set in QUERY_SETS[dataset]:
            table = read_features(paths.features_file(backbone, dataset, query_set, features_root))
            present = set(table.ids.tolist())
            selected = [q for q in wanted if q in present]
            if not selected:
                continue
            table = table.select(selected)
            vectors = table.vectors(recipe)
            targets = gallery_targets(table.labels["mesh_id"].tolist(), gallery.mesh_ids)
            raw = score_meshes(vectors, gallery.views.float().cpu().numpy(), zero_shot_cfg["scoring"], zero_shot_cfg["alpha"])
            sets.append(ValSet(
                name=f"{dataset}_{query_set}",
                dataset=dataset,
                features=torch.from_numpy(vectors).to(device, torch.float16),
                targets=targets.to(device),
                categories=torch.tensor([category_index[c] for c in table.labels["category"]], device=device),
                zero_shot_scores=torch.from_numpy(raw).to(device),
                zero_shot_r1=metrics.recall_at_k(metrics.correct_ranks(raw, targets.numpy()), 1),
                query_ids=table.labels["query_id"], mesh_ids=table.labels["mesh_id"], category_names=table.labels["category"],
            ))
    return sets


def _ranks(s: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Rank of the correct mesh (1 = first); ties count against it, as in metrics.correct_ranks."""
    return (s >= s.gather(1, targets[:, None])).sum(dim=1)


# Mix weights whose per-photo ranks every validation records (`validate`, `ranks_out`), so that a
# cross-validation can compare mix weights afterwards: 0 = zero-shot, 1 = trained alone.
RECORDED_MIX_WEIGHTS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)


@torch.no_grad()
def validate(heads: EmbeddingHeads, val_sets: list[ValSet], galleries: dict[str, Gallery], alpha: float,
             mix_weight: float = 1.0, ranks_out: dict | None = None) -> dict[str, dict]:
    """Embed the val photos and the full galleries with the current heads; return the metrics of each val set.

    The metrics use the final score (mix_weight × trained + (1 − mix_weight) × zero-shot);
    "trained_R@1" is the R@1 of the trained score alone.

    Args:
        ranks_out: If given, it receives {val set name: int16 (len(RECORDED_MIX_WEIGHTS), Q)}: the
            rank of the correct mesh of every photo for each recorded mix weight.
    """
    heads.eval()
    embedded = {d: heads.embed_meshes(g.views) for d, g in galleries.items()}  # all views, no dropout
    results = {}
    for val in val_sets:
        views, meshes = embedded[val.dataset]
        trained = scores(heads.embed_queries(val.features), views, meshes, alpha)
        s = mix_weight * trained + (1.0 - mix_weight) * val.zero_shot_scores
        ranks = _ranks(s, val.targets)
        first_category = galleries[val.dataset].categories[s.argmax(dim=1)]
        results[val.name] = {
            "photos": len(ranks),
            **{f"R@{k}": float((ranks <= k).float().mean()) for k in (1, 5, 10)},
            "cat@1": float((first_category == val.categories).float().mean()),
            "trained_R@1": float((_ranks(trained, val.targets) <= 1).float().mean()),
            "zero_shot_R@1": val.zero_shot_r1,
        }
        if ranks_out is not None:
            ranks_out[val.name] = torch.stack([
                _ranks(w * trained + (1.0 - w) * val.zero_shot_scores, val.targets) for w in RECORDED_MIX_WEIGHTS
            ]).to(torch.int16).cpu().numpy()
    heads.train()
    return results


def selection_value(results: dict[str, dict]) -> float:
    """The value that selects the best epoch: the mean R@1 over the val query sets."""
    return float(np.mean([r["R@1"] for r in results.values()]))


def parameter_groups(heads: EmbeddingHeads, weight_decay: float) -> list[dict]:
    """Weight decay for the weight matrices only; none for biases and the rating vector (1-D parameters)."""
    decay = [p for p in heads.parameters() if p.ndim >= 2]
    no_decay = [p for p in heads.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def learning_rate_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Factor of the base learning rate: linear rise during the warm-up, then a cosine fall to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_step(heads: EmbeddingHeads, optimizer, scheduler, query_features: torch.Tensor, targets: torch.Tensor,
               query_categories: torch.Tensor, gallery: Gallery, train_cfg: dict, generator: torch.Generator) -> dict:
    """One update of the heads (steps 3 to 8 of the module description); return the loss values and the batch R@1."""
    heads_cfg, loss_cfg = train_cfg["heads"], train_cfg["loss"]
    q = heads.embed_queries(query_features)
    mask = random_view_mask(len(gallery.mesh_ids), gallery.views.shape[1], heads_cfg["view_dropout_min_views"],
                            generator=generator, device=gallery.views.device)
    views, meshes = heads.embed_meshes(gallery.views, mask)
    s = scores(q, views, meshes, train_cfg["scoring"]["alpha"])
    loss = gallery_info_nce(s, targets, query_categories, gallery.categories, loss_cfg["temperature"], loss_cfg["category_weight"])
    optimizer.zero_grad(set_to_none=True)
    loss["loss"].backward()
    optimizer.step()
    scheduler.step()
    return {**{k: float(v) for k, v in loss.items()}, "batch_R@1": float((s.argmax(dim=1) == targets).float().mean())}


def train_epoch(heads: EmbeddingHeads, extractor: FeatureExtractor, loader: DataLoader, optimizer, scheduler,
                gallery: Gallery, train_cfg: dict, background_rgb, generator: torch.Generator, device,
                desc: str = "train") -> dict:
    """One pass over the training photos; return the mean loss values and the speed figures.

    A progress bar (on stderr, so it appears at once also through a pipe) shows the steps, the
    speed, the time left in the epoch, and the running mean loss.
    """
    sums, steps, images = {}, 0, 0
    wait = compute = 0.0
    needs_masks = extractor.recipe != "cls"
    mark = time.perf_counter()
    progress = tqdm(loader, desc=desc, unit="step", leave=False, dynamic_ncols=True)
    for batch in progress:
        loaded = time.perf_counter()
        wait += loaded - mark  # time spent waiting for the data loader
        images_np = batch["image"].numpy()
        features = torch.from_numpy(extractor.extract(images_np, object_masks(images_np, background_rgb) if needs_masks else None)).to(device)
        values = train_step(heads, optimizer, scheduler, features, batch["mesh"].to(device), batch["category"].to(device),
                            gallery, train_cfg, generator)
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + value
        steps += 1
        images += len(images_np)
        progress.set_postfix(loss=f"{sums['loss'] / steps:.3f}")
        mark = time.perf_counter()
        compute += mark - loaded
    means = {key: value / max(steps, 1) for key, value in sums.items()}
    total = wait + compute
    return {**means, "steps": steps, "images": images, "seconds": round(total, 1),
            "images_per_second": round(images / total, 1) if total else 0.0,
            "loader_wait_share": round(wait / total, 3) if total else 0.0}
