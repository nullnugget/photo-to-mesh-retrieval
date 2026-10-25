"""Re-ranking method 1: a patch-by-patch comparison of a photo and each listed mesh (no training).

Stage 1 compares one vector per image (the class token), which summarizes the whole object. Method 1
compares the 196 patch vectors (one per 16 x 16 square), so a detail that decides between two similar
meshes (an armrest, a crossbar, the shape of a leg) can count:

1. For one photo and one view of a candidate mesh: the cosine similarity of every photo patch with
   every view patch (a matrix, photo patches x view patches). Only the patches on the object count.
2. Each photo patch takes its best match in the view; the mean of these best matches. The same the
   other way (each view patch takes its best match in the photo). The patch score of the view is the
   mean of both directions. Both directions, because one direction alone rewards a view that only
   contains the photo's parts (a large mesh with extra parts would match every photo patch).
3. The patch score of a mesh is the best of its views (the views with the highest stage-1 view score).
4. Final score = stage-1 score + weight x patch score; the weight is chosen on the cross-validation folds.

Example (3 object patches each):

    photo patches: leg, seat, armrest      view patches: leg', seat', back'
    photo -> view: leg→leg' 0.82, seat→seat' 0.77, armrest→back' 0.41   mean 0.67
    view -> photo: leg'→leg 0.82, seat'→seat 0.77, back'→armrest 0.41   mean 0.67
    a candidate whose view has an armrest scores higher on the third patch
"""

import numpy as np
import torch
import torch.nn.functional as F

NO_MATCH = -2.0  # below every cosine similarity: a background patch never is the best match


def patch_scores(photo: torch.Tensor, photo_object: torch.Tensor, views: torch.Tensor, view_object: torch.Tensor) -> torch.Tensor:
    """Patch score of every view for every photo.

    Args:
        photo: (B, P, d) patch vectors of B photos (any length; scaled to length 1 here).
        photo_object: bool (B, P), True on the object.
        views: (B, C, V, Q, d) patch vectors of V views of C candidates of each photo.
        view_object: bool (B, C, V, Q).

    Returns:
        float32 (B, C, V): the mean of the two best-match directions. An image without any object
        patch uses all its patches.
    """
    photo_object = photo_object | ~photo_object.any(-1, keepdim=True)
    view_object = view_object | ~view_object.any(-1, keepdim=True)
    photo = F.normalize(photo.float(), dim=-1).to(views.dtype)
    views = F.normalize(views.float(), dim=-1).to(photo.dtype)
    sim = torch.einsum("bpd,bcvqd->bcvpq", photo, views).float()  # (B, C, V, P, Q)
    photo_mask = photo_object[:, None, None, :, None]  # (B, 1, 1, P, 1)
    view_mask = view_object[:, :, :, None, :]  # (B, C, V, 1, Q)
    best_in_view = sim.masked_fill(~view_mask, NO_MATCH).amax(-1)  # (B, C, V, P): each photo patch's best match
    best_in_photo = sim.masked_fill(~photo_mask, NO_MATCH).amax(-2)  # (B, C, V, Q): each view patch's best match
    p = photo_object[:, None, None, :].float()
    q = view_object.float()
    photo_to_view = (best_in_view * p).sum(-1) / p.sum(-1)
    view_to_photo = (best_in_photo * q).sum(-1) / q.sum(-1)
    return (photo_to_view + view_to_photo) / 2


def reranked_ranks(stage1_rank: np.ndarray, top_index: np.ndarray, truth: np.ndarray, top_score: np.ndarray,
                   patch: np.ndarray, weight: float, top: int) -> np.ndarray:
    """Rank of the correct mesh after re-ordering the first `top` candidates.

    Args:
        stage1_rank: (N,) rank of the correct mesh in stage 1.
        top_index: (N, >= top) the listed meshes; truth: (N,) the correct mesh.
        top_score: (N, >= top) their stage-1 scores. patch: (N, >= top) their patch scores.
        weight: The weight of the patch score.

    A photo whose correct mesh is not in the first `top` keeps its stage-1 rank (a re-ranker cannot
    bring it in). A tie counts against the correct mesh, as in `metrics.correct_ranks`.
    """
    match = top_index[:, :top] == truth[:, None]
    inside = match.any(1)
    final = top_score[:, :top] + weight * patch[:, :top]
    truth_final = final[np.arange(len(final)), match.argmax(1)]
    new = (final >= truth_final[:, None]).sum(1)
    return np.where(inside, new, np.maximum(stage1_rank, top + 1))
