"""Stage-1 candidate lists for the re-ranking: the first meshes of every photo, "out of fold".

A re-ranker must be measured (and later trained) on lists that a stage-1 model made for photos it
did not train on; otherwise the correct mesh is nearly always first and nothing is left to re-order.
The cross-validation gives this: each non-test photo of a split is a val photo of exactly one fold
model. So fold model k scores the val photos of fold k, against the full gallery of the photo's own
dataset, exactly as the validation does (trained score mixed with the zero-shot score):

    stage-1 score = mix * trained score + (1 - mix) * zero-shot score

Per photo the list keeps:
- the `top` meshes with the highest stage-1 score, and their scores;
- the trained score of every view of each listed mesh (the re-ranker compares the best views only);
- the rank and the score of the correct mesh (rank 1 = first).

File (one per query set): `data/derived/rerank/<tag>/<dataset>_<set>_candidates.npz`, with the gallery
mesh ids, so that `top` indexes into `gallery`.
"""

import json
from pathlib import Path

import numpy as np
import torch

from src import paths, runs
from src.models.embedding_heads import scores
from src.models.feature_store import read_features
from src.retrieval.evaluate import score_meshes
from src.retrieval.index import load_heads
from src.train import cross_validation

RERANK_DIR = paths.DERIVED_DIR / "rerank"
QUERY_SETS = {"pix3d": ("queries_reference", "queries_sam"), "abo": ("queries_sam",)}
BLOCK = 512  # photos per scoring block (a block of 512 x 3,010 meshes x 25 views fits on a 6 GB GPU)


def candidates_file(tag: str, dataset: str, query_set: str, root: Path = RERANK_DIR) -> Path:
    return root / tag / f"{dataset}_{query_set}_candidates.npz"


def fold_models(split: str, tag: str, datasets: list[str], backbone: str, folds: int) -> list[Path]:
    """Return the last-epoch checkpoint of each fold run of a cross-validation, fold 1 first."""
    run_dirs = cross_validation.find_fold_runs("_".join(datasets), backbone, split, folds, tag)
    return [paths.CHECKPOINTS_DIR / run_dir.name / "embedding_heads_last.pt" for run_dir in run_dirs]


@torch.no_grad()
def score_photos(heads, alpha: float, mix: float, zero_shot: dict, queries: np.ndarray, gallery: np.ndarray,
                 truth: np.ndarray, top: int, device: str) -> dict[str, np.ndarray]:
    """Stage-1 lists of a block of photos.

    Args:
        queries: Raw query features (Q, D), unit length. gallery: raw view features (M, V, D), unit length.
        truth: Index of the correct mesh of each photo (Q,).
    """
    views16 = torch.from_numpy(gallery.astype(np.float16)).to(device)
    view_emb, mesh_emb = heads.embed_meshes(views16)
    out: dict[str, list] = {"top": [], "top_score": [], "view_score": [], "rank": [], "truth_score": []}
    for start in range(0, len(queries), BLOCK):
        q_raw = queries[start:start + BLOCK]
        zs = torch.from_numpy(score_meshes(q_raw, gallery, zero_shot["scoring"], zero_shot["alpha"])).to(device)
        q = heads.embed_queries(torch.from_numpy(q_raw).to(device, torch.float16))
        s = mix * scores(q, view_emb, mesh_emb, alpha).float() + (1 - mix) * zs
        t = torch.from_numpy(truth[start:start + BLOCK]).to(device)[:, None]
        truth_score = s.gather(1, t)
        top_score, top_index = s.topk(top, dim=1)
        out["rank"].append((s >= truth_score).sum(1))  # a tie counts against the correct mesh, as in metrics.correct_ranks
        out["truth_score"].append(truth_score[:, 0])
        out["top"].append(top_index); out["top_score"].append(top_score)
        out["view_score"].append(torch.cat([  # 10 listed meshes at a time: (512, 50, 25, 768) at once is 1 GB or more
            torch.einsum("qd,qlvd->qlv", q.half(), view_emb[top_index[:, i:i + 10]].half())
            for i in range(0, top, 10)], dim=1))
    return {key: torch.cat(values).cpu().numpy() for key, values in out.items()}


def build(cfg: dict, split: str, tag: str, datasets: list[str], folds: int, device: str = "cuda",
          root: Path = RERANK_DIR) -> dict:
    """Write the out-of-fold candidate lists of every query set; return a summary (R@1, R@10 of each set).

    Every val photo of every fold gets a list. Photos outside the feature file (no SAM mask) are skipped.
    """
    backbone, top, mix = cfg["backbone"]["name"], cfg["rerank"]["candidates"]["top"], cfg["train"]["scoring"]["mix_weight"]
    zero_shot = cfg["backbone"]["zero_shot"]
    checkpoints = fold_models(split, tag, datasets, backbone, folds)
    rows: dict[tuple, dict[str, list]] = {}
    galleries = {}
    for dataset in datasets:
        g = read_features(paths.features_file(backbone, dataset, "renders"))
        galleries[dataset] = (g.ids, g.vectors(cfg["backbone"]["feature_recipe"]))
    for k, checkpoint in enumerate(checkpoints, start=1):
        ck = runs.load_checkpoint(checkpoint)
        heads = load_heads(ck, device)
        alpha = ck["meta"]["config"]["train"]["scoring"]["alpha"]
        for dataset in datasets:
            ids, gallery = galleries[dataset]
            index = {m: i for i, m in enumerate(ids)}
            val = json.loads(paths.split_file(dataset, f"{split}_fold{k}").read_text())["parts"]["val"]
            for query_set in QUERY_SETS[dataset]:
                table = read_features(paths.features_file(backbone, dataset, query_set))
                present = set(table.ids.tolist())
                table = table.select([q for q in val if q in present])
                truth = np.array([index[m] for m in table.labels["mesh_id"]])
                lists = score_photos(heads, alpha, mix, zero_shot, table.vectors(cfg["backbone"]["feature_recipe"]),
                                     gallery, truth, top, device)
                r = rows.setdefault((dataset, query_set), {"query_id": [], "fold": [], "truth": [], **{key: [] for key in lists}})
                r["query_id"].append(table.ids); r["fold"].append(np.full(len(truth), k)); r["truth"].append(truth)
                for key, values in lists.items():
                    r[key].append(values)
        print(f"fold {k}: {checkpoint.parent.name}", flush=True)

    summary = {}
    for (dataset, query_set), r in rows.items():
        arrays = {key: np.concatenate(values) for key, values in r.items()}
        target = candidates_file(tag, dataset, query_set, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, gallery=galleries[dataset][0], mix=mix, **arrays)
        rank = arrays["rank"]
        summary[f"{dataset}_{query_set}"] = {"photos": len(rank), "R@1": float((rank <= 1).mean()),
                                            "R@10": float((rank <= 10).mean()), f"R@{top}": float((rank <= top).mean())}
        print(f"{dataset}_{query_set}: {len(rank)} photos  R@1 {100 * (rank <= 1).mean():.1f}  R@10 {100 * (rank <= 10).mean():.1f}  "
              f"R@{top} {100 * (rank <= top).mean():.1f} -> {target.relative_to(paths.ROOT)}", flush=True)
    return summary


def read_candidates(tag: str, dataset: str, query_set: str, root: Path = RERANK_DIR) -> dict[str, np.ndarray]:
    with np.load(candidates_file(tag, dataset, query_set, root), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
