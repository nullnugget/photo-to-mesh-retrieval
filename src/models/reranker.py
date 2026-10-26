"""Re-ranking method 2: a learned pair scorer that corrects the score of each listed mesh.

Method 1 compares the patches of a photo and of a view with a fixed rule (each patch takes its best
match, both directions). Method 2 learns the comparison. For one (photo, view) pair:

1. Tokens: the object patches of each image (at most `tokens` of them; 128 numbers each, from the
   patch store). Each token is made 256 numbers long (a linear layer), and three learned vectors are
   added: the vector of its grid row, of its grid column (28 x 28 at 448 pixels), and of its side
   (photo or render). So the model knows where each patch is and which image it comes from.
2. Blocks (2): cross-attention in both directions with one set of weights (photo patches ask, view
   patches answer; then view patches ask, photo patches answer), then a feed-forward layer. Padding
   tokens are never attended to.
3. The mean over the real tokens of each side: p (photo) and v (view).
4. [p, v, |p - v|, p x v] -> 256 -> 1: the correction r_v of this view. The last layer starts at 0, so
   r_v = 0 for every pair before training.

The correction of a mesh is the best of its views; the score that orders the list is
    stage-1 score + method-1 weight x method-1 score + r     (variant "residual_method1")
    stage-1 score + r                                       (variant "stage1")
so at step 0 the order is that of method 1 or that of stage 1.
"""

import torch
import torch.nn.functional as F
from torch import nn

VARIANTS = ("residual_method1", "stage1")
PHOTO, RENDER = 0, 1  # rows of the side table


class CrossAttention(nn.Module):
    """Multi-head attention of `queries` over `keys_values`; key padding masked out."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = (nn.Linear(width, width) for _ in range(4))

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """queries (N, Lq, W), keys_values (N, Lk, W), key_mask bool (N, Lk), True = a real token."""
        n, lq, w = queries.shape
        split = lambda x: x.view(n, -1, self.heads, w // self.heads).transpose(1, 2)  # (N, heads, L, W / heads)
        out = F.scaled_dot_product_attention(split(self.q(queries)), split(self.k(keys_values)), split(self.v(keys_values)),
                                             attn_mask=key_mask[:, None, None, :])
        return self.out(out.transpose(1, 2).reshape(n, lq, w))


class CrossBlock(nn.Module):
    """One block: cross-attention both ways (shared weights), then a feed-forward layer on each side."""

    def __init__(self, width: int, heads: int, ffn: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = CrossAttention(width, heads)
        self.feed_forward = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))

    def forward(self, p, p_mask, v, v_mask):
        pn, vn = self.norm1(p), self.norm1(v)
        p, v = p + self.attention(pn, vn, v_mask), v + self.attention(vn, pn, p_mask)  # both from the old values
        return p + self.feed_forward(self.norm2(p)), v + self.feed_forward(self.norm2(v))


class PairScorer(nn.Module):
    """The correction r_v of one (photo, view) pair (module description)."""

    def __init__(self, in_dim: int = 128, width: int = 256, heads: int = 4, blocks: int = 2, ffn: int = 1024, grid: int = 28):
        super().__init__()
        self.grid = grid
        self.project = nn.Linear(in_dim, width)
        self.row, self.column, self.side = nn.Embedding(grid, width), nn.Embedding(grid, width), nn.Embedding(2, width)
        for table in (self.row, self.column, self.side):
            nn.init.normal_(table.weight, std=0.02)
        self.blocks = nn.ModuleList(CrossBlock(width, heads, ffn) for _ in range(blocks))
        self.norm = nn.LayerNorm(width)
        self.score = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.score[-1].weight)  # r_v = 0 before training: the order starts as that of the base score
        nn.init.zeros_(self.score[-1].bias)

    def tokens(self, x: torch.Tensor, position: torch.Tensor, side: int) -> torch.Tensor:
        """(N, L, in_dim) patch vectors and their patch numbers (N, L) -> (N, L, width) tokens."""
        side_index = torch.full_like(position, side)
        return self.project(x) + self.row(position // self.grid) + self.column(position % self.grid) + self.side(side_index)

    def forward(self, photo, photo_position, photo_mask, view, view_position, view_mask) -> torch.Tensor:
        """N pairs: patch vectors (N, L, in_dim), patch numbers (N, L), real-token masks (N, L) -> r_v (N,)."""
        p, v = self.tokens(photo, photo_position, PHOTO), self.tokens(view, view_position, RENDER)
        for block in self.blocks:
            p, v = block(p, photo_mask, v, view_mask)
        p, v = _masked_mean(self.norm(p), photo_mask), _masked_mean(self.norm(v), view_mask)
        return self.score(torch.cat([p, v, (p - v).abs(), p * v], dim=-1)).squeeze(-1)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)[..., None]
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


def select_tokens(vectors: torch.Tensor, on_object: torch.Tensor, count: int,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose at most `count` object patches of each image.

    Args:
        vectors: (n, P, d) patch vectors of n images. on_object: bool (n, P), the object patches (the same
            rule as method 1: object share at least `rerank.patches.object_share`).
        generator: the random choice among the object patches when an image has more than `count`
            (a fixed seed in evaluation, so that the same patches are taken each time).

    Returns:
        tokens (n, count, d), patch numbers (n, count), real-token mask (n, count). An image without any
        object patch uses `count` of all its patches.
    """
    on_object = on_object | ~on_object.any(1, keepdim=True)
    device = generator.device if generator is not None else on_object.device
    noise = torch.rand(on_object.shape, generator=generator, device=device).to(on_object.device)
    position = (on_object.float() * 2 + noise).topk(count, dim=1).indices.sort(dim=1).values  # object patches first
    mask = on_object.gather(1, position)
    tokens = vectors.gather(1, position[..., None].expand(-1, -1, vectors.shape[-1]))
    return tokens, position, mask


def final_scores(variant: str, stage1: torch.Tensor, method1: torch.Tensor, correction: torch.Tensor,
                 method1_weight: float) -> torch.Tensor:
    """The score that orders the list (module description); all (B, C)."""
    if variant == "residual_method1":
        return stage1 + method1_weight * method1 + correction
    if variant == "stage1":
        return stage1 + correction
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
