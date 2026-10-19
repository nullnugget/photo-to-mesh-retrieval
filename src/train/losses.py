"""The training loss: full-gallery InfoNCE plus a category term.

For each query (a training photo), the combined scores against every mesh of the training gallery
become probabilities with a softmax at temperature tau:

    p(m | query) = exp(s(query, m) / tau) / sum over all meshes k of exp(s(query, k) / tau)

- Instance loss: L_inst = −ln p(correct mesh | query). Every query competes against every wrong
  mesh of the gallery, also every look-alike of its category ("full gallery"). This is InfoNCE: a
  cross-entropy over similarity scores, where the "classes" are the mesh embeddings that the heads
  calculate from the renders, not fixed weights. So the loss pulls the query towards its mesh and
  pushes it away from all other meshes.
- Category loss: L_cat = −ln (sum of p(m | query) over the meshes of the correct category). It asks
  "is it at least the right category?". It uses the same probabilities, so it needs no extra network,
  and it works for any set of categories.
- Total: L = mean over the batch of (L_inst + lambda × L_cat).

Why the full gallery and not the meshes of the batch (in-batch InfoNCE): with the meshes of the
batch only, a mesh with two photos in the batch appears twice and one copy counts as wrong (a false
negative), and the hardest wrong mesh is usually not in the batch. Here, each mesh is one column of
the score matrix (`gallery_targets` checks this), so neither problem can occur.

The gradient of L_inst with respect to the score of mesh m is (p(m) − [m is correct]) / tau: the
correct mesh is pulled up, and each wrong mesh is pushed down in proportion to its probability, so
the look-alikes that compete most get the strongest push.
"""

import torch


def gallery_targets(query_mesh_ids, gallery_mesh_ids) -> torch.Tensor:
    """Return the column of each query's correct mesh in the gallery: int64 (queries,).

    Raises:
        ValueError: If a mesh appears twice in the gallery (it would count once as right and once
            as wrong for the same query).
        KeyError: If the correct mesh of a query is not in the gallery.
    """
    column = {}
    for index, mesh_id in enumerate(gallery_mesh_ids):
        if mesh_id in column:
            raise ValueError(f"mesh {mesh_id!r} appears twice in the gallery")
        column[mesh_id] = index
    missing = [mesh_id for mesh_id in query_mesh_ids if mesh_id not in column]
    if missing:
        raise KeyError(f"{len(missing)} correct meshes are not in the gallery, for example {missing[:3]}")
    return torch.tensor([column[mesh_id] for mesh_id in query_mesh_ids], dtype=torch.long)


def gallery_info_nce(scores: torch.Tensor, targets: torch.Tensor, query_categories: torch.Tensor,
                     mesh_categories: torch.Tensor, temperature: float, category_weight: float) -> dict[str, torch.Tensor]:
    """Return the batch loss and its parts.

    Args:
        scores: Combined scores (queries, meshes) of the batch against the whole gallery.
        targets: int64 (queries,), the column of each query's correct mesh (`gallery_targets`).
        query_categories: int64 (queries,), the category number of each query.
        mesh_categories: int64 (meshes,), the category number of each mesh.
        temperature: tau; a smaller tau makes the probabilities sharper.
        category_weight: lambda, the weight of the category loss (0 = instance loss only).

    Returns:
        {"loss": batch mean of L_inst + lambda × L_cat, "instance": batch mean of L_inst,
         "category": batch mean of L_cat}. Only "loss" is meant for backward().
    """
    log_p = torch.log_softmax(scores / temperature, dim=1)  # ln p(m | query), stable for large scores
    instance = -log_p.gather(1, targets[:, None]).squeeze(1)
    same_category = query_categories[:, None] == mesh_categories[None, :]  # (queries, meshes)
    # ln of the summed probability of the correct category: logsumexp over those meshes only.
    category = -torch.logsumexp(log_p.masked_fill(~same_category, float("-inf")), dim=1)
    return {
        "loss": (instance + category_weight * category).mean(),
        "instance": instance.mean(),
        "category": category.mean(),
    }
