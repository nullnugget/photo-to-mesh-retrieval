"""The embedding heads: three small trainable networks on top of the frozen backbone features.

The backbone (DINOv3) describes a photo and a grey render of the same mesh in different places of
its feature space: colour, texture, lighting, and photo or render style all count. The embedding
heads learn to translate both into one shared space where only the mesh counts.

    query feature (768) ──► QueryProjector ──► query embedding q (256, length 1)
    view features (25 × 768) ──► ViewProjector ──► view embeddings e (25 × 256, length 1)
                                                        │
                                                        ▼
                                                 ViewAggregator ──► mesh embedding g (256, length 1)

- QueryProjector and ViewProjector have the same structure (Linear 768→512, GELU, Linear 512→256,
  scale to length 1) but separate weights: photos and grey renders are like two languages, so each
  gets its own translator.
- ViewAggregator combines the view embeddings of one mesh into one mesh embedding by attention
  pooling: a learned vector rates each view, a softmax turns the ratings into weights, and the
  weighted mean of the view embeddings, scaled to length 1, is the mesh embedding. The mean is taken
  of the view embeddings themselves, so the mesh embedding stays in the same space as the view and
  query embeddings (the combined score adds a view similarity and a mesh similarity). The rating
  vector starts at zero, so training starts from the plain mean of the views.

Scores (the same definitions as the zero-shot search in `src/retrieval/search.py`, but in PyTorch,
so that the gradients reach the heads):

- view score: the highest similarity between q and the view embeddings of a mesh;
- mesh score: the similarity between q and the mesh embedding;
- combined score: alpha × view score + (1 − alpha) × mesh score.

View dropout: during training, the aggregator sees a random 12 to 25 views of each mesh
(`random_view_mask`), so that it does not depend on any single view.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

AGGREGATORS = ("attention", "mean")


class Projector(nn.Module):
    """Linear in→hidden, GELU, Linear hidden→out, then scale to length 1."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features (..., in_dim) -> unit-length embeddings (..., out_dim)."""
        return F.normalize(self.layers(features.float()), dim=-1)


class QueryProjector(Projector):
    """Turns a query feature (a photo) into a query embedding."""


class ViewProjector(Projector):
    """Turns a view feature (one render) into a view embedding."""


class ViewAggregator(nn.Module):
    """Combines the view embeddings of each mesh into one mesh embedding.

    "attention": rating of view v = (K e_v) · w / sqrt(dim), with a learned key layer K and a learned
    vector w; weights = softmax of the ratings over the views that the mask keeps; mesh embedding =
    the weighted mean of the view embeddings, scaled to length 1. w starts at zero, so all views start
    with equal weight. "mean": equal weights, no parameters.
    """

    def __init__(self, dim: int, kind: str = "attention"):
        super().__init__()
        if kind not in AGGREGATORS:
            raise ValueError(f"kind must be one of {AGGREGATORS}, got {kind!r}")
        self.kind = kind
        if kind == "attention":
            self.key = nn.Linear(dim, dim)
            self.rating = nn.Parameter(torch.zeros(dim))  # zero: the training starts from the plain mean
            self.scale = 1.0 / math.sqrt(dim)

    def weights(self, views: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the weight of every view: (meshes, views), each row adds up to 1.

        Args:
            views: View embeddings (meshes, views, dim).
            mask: bool (meshes, views), True = the view takes part; None = all views.
        """
        if self.kind == "attention":
            ratings = (self.key(views) @ self.rating) * self.scale  # (meshes, views)
        else:
            ratings = torch.zeros(views.shape[:2], device=views.device)
        if mask is not None:
            ratings = ratings.masked_fill(~mask, float("-inf"))  # a dropped view gets weight 0
        return torch.softmax(ratings, dim=1)

    def forward(self, views: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """views (meshes, views, dim) -> unit-length mesh embeddings (meshes, dim)."""
        pooled = (self.weights(views, mask)[..., None] * views).sum(dim=1)
        return F.normalize(pooled, dim=-1)


class EmbeddingHeads(nn.Module):
    """The three heads together. They only work as a set, so they are trained and saved together."""

    def __init__(self, feature_dim: int, hidden_dim: int = 512, embedding_dim: int = 256, aggregator: str = "attention"):
        super().__init__()
        self.query_projector = QueryProjector(feature_dim, hidden_dim, embedding_dim)
        self.view_projector = ViewProjector(feature_dim, hidden_dim, embedding_dim)
        self.view_aggregator = ViewAggregator(embedding_dim, aggregator)

    @classmethod
    def from_config(cls, heads_cfg: dict, feature_dim: int) -> "EmbeddingHeads":
        """Build the heads from `train.heads` and the feature length of the backbone."""
        return cls(feature_dim, heads_cfg["hidden_dim"], heads_cfg["embedding_dim"], heads_cfg["aggregator"])

    def embed_queries(self, query_features: torch.Tensor) -> torch.Tensor:
        """(queries, feature_dim) -> query embeddings (queries, embedding_dim)."""
        return self.query_projector(query_features)

    def embed_meshes(self, view_features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(meshes, views, feature_dim) -> (view embeddings (meshes, views, dim), mesh embeddings (meshes, dim))."""
        views = self.view_projector(view_features)
        return views, self.view_aggregator(views, mask)


def parameter_counts(heads: EmbeddingHeads) -> dict[str, int]:
    """Number of trainable parameters of each head and in total."""
    counts = {name: sum(p.numel() for p in getattr(heads, name).parameters()) for name in ("query_projector", "view_projector", "view_aggregator")}
    counts["total"] = sum(counts.values())
    return counts


def scores(queries: torch.Tensor, views: torch.Tensor, meshes: torch.Tensor, alpha: float) -> torch.Tensor:
    """Combined score of every mesh for every query: (queries, meshes).

    Args:
        queries: Query embeddings (Q, D), unit length.
        views: View embeddings (M, V, D), unit length. All views count for the view score.
        meshes: Mesh embeddings (M, D), unit length.
        alpha: Weight of the view score (1.0 = view score only, 0.0 = mesh score only).
    """
    view_score = torch.einsum("qd,mvd->qmv", queries, views).amax(dim=2)  # best view of each mesh (hard maximum)
    mesh_score = queries @ meshes.T
    return alpha * view_score + (1.0 - alpha) * mesh_score


def random_view_mask(meshes: int, views: int, min_views: int, generator: torch.Generator | None = None,
                     device: str | torch.device = "cpu") -> torch.Tensor:
    """Return a random view mask for training: bool (meshes, views).

    Each mesh keeps a random number of views between min_views and views (both included), chosen
    at random positions.
    """
    if not 1 <= min_views <= views:
        raise ValueError(f"min_views must be between 1 and {views}, got {min_views}")
    keep = torch.randint(min_views, views + 1, (meshes, 1), generator=generator)  # views to keep per mesh
    order = torch.rand(meshes, views, generator=generator).argsort(dim=1).argsort(dim=1)  # a random rank per view
    return (order < keep).to(device)
