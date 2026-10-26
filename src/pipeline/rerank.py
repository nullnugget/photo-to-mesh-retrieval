"""The re-ranking step (stage 2), measured on the cross-validation folds of a stage-1 recipe.

Stages:
1. candidates: the out-of-fold stage-1 lists of every val photo (`src/retrieval/candidates.py`).
2. method1: the patch score (`src/retrieval/patch_rerank.py`) of the first `rerank.method1.top`
   candidates of every photo, from the patch store (`src/models/patch_store.py`), for each number of
   patch dimensions in `rerank.method1.dims` (the first d of the stored numbers).
3. report: R@1 after the re-ordering for each weight, and the weight chosen fairly:
   - one weight for every photo of every source (one model for all sources), chosen on the pooled photos of the other
     folds and applied to the photos of the held-out fold ("leave one fold out"), so that no photo
     helps to choose the weight that it is measured with;
   - the paired difference against stage 1 over meshes (bootstrap), as `09_cross_validation.py --compare`.

4. gallery: the patch comparison as a stage 1, against every mesh (a sample of photos; `gallery_test`).
5. dimcheck: the cost of the PCA reduction, raw 768 numbers against the stored reduction (a sample; `dim_check`).
6. sizecheck: method 1 with the patches of larger images (224, 336, 448 pixels; a sample; `size_check`).

Outputs: `data/derived/rerank/<tag>/` (`<dataset>_<set>_candidates.npz`, `<dataset>_<set>_method1.npz`,
`method1_v<views>_report.json`, `gallery_report.json`, `dimcheck_report.json`, `sizecheck_report.json`; the method-1 files carry
the number of views per candidate, `_v<views>`, so that runs with different view counts stay side by side).
A list length other than 10 (`rerank.method1.top`) adds `_top<top>` to the method-1 files and report.
The patch store is that of `rerank.patches.size`; at another size than the main one, the outputs of the method1,
report, gallery, and dimcheck stages carry its suffix (for example `method1_v5_448_report.json`, `dimcheck_448_report.json`).
"""

import json

import numpy as np
import torch
from tqdm import tqdm

from src import paths
from src.models.patch_store import patch_files, read_patches, store_name
from src.retrieval import candidates as cand
from src.retrieval.metrics import paired_difference
from src.retrieval.patch_rerank import patch_scores, reranked_ranks

VIEWS_PER_BLOCK = 160  # photos per block = 160 // views at 196 patches (fewer at more patches, by their square):
                       # 32 photos x 10 candidates x 5 views x 196 x 196 similarities
                       # is about 1.5 GB on the GPU
SIMILARITIES_PER_CALL = 16 * 50 * 196 * 196  # patch_scores calls of about 0.12 GB float32 similarities: 16 photos
                                              # at 196 patches, 3 at 441, 1 at 784


def store_of(cfg: dict) -> tuple[str, str]:
    """Return (store name, file suffix) of the patch store of `rerank.patches.size` ("" and the plain name at the
    main size; "_448" at 448, so that the outputs of both sizes stay side by side)."""
    size, main = cfg["rerank"]["patches"]["size"], cfg["data"]["image_size"]
    return store_name(cfg["backbone"]["name"], size, main), paths.size_suffix(size, main)


DEFAULT_TOP = 10  # list length without a mark in the file names (the files made before the length was a choice)


def top_mark(top: int) -> str:
    """Return the file-name mark of a list length: none for the first 10, else "_top<top>" (for example "_top20").
    So runs of other list lengths stay side by side."""
    return "" if top == DEFAULT_TOP else f"_top{top}"


def method1_file(tag: str, dataset: str, query_set: str, views: int, suffix: str = "", top: int = DEFAULT_TOP):
    return cand.RERANK_DIR / tag / f"{dataset}_{query_set}_method1_v{views}{suffix}{top_mark(top)}.npz"


def best_views(view_score: np.ndarray, top: int, views: int) -> np.ndarray:
    """(N, >= top, V) stage-1 view scores -> (N, top, views) indices of the best views of each candidate."""
    return np.argsort(-view_score[:, :top].astype(np.float32), axis=-1)[..., :views]


def method1_scores(cfg: dict, tag: str, dataset: str, query_set: str, device: str = "cuda") -> dict[str, np.ndarray]:
    """Patch scores of the first candidates of every photo of one set: {f"patch_{d}": (N, top)} and the views used."""
    m1, (store, _) = cfg["rerank"]["method1"], store_of(cfg)
    threshold = round(cfg["rerank"]["patches"]["object_share"] * 255)
    lists = cand.read_candidates(tag, dataset, query_set)
    photos = read_patches(patch_files(store, dataset, query_set))
    renders = read_patches(patch_files(store, dataset, "renders"))
    if not np.array_equal(renders.ids, lists["gallery"]):
        raise ValueError("the render rows of the patch store and the gallery of the candidate lists differ")
    row = photos.row_index()
    photo_rows = np.array([row[q] for q in lists["query_id"]])
    top, dims = m1["top"], m1["dims"]
    meshes = lists["top"][:, :top]
    views = best_views(lists["view_score"], top, m1["views"])  # (N, top, V)
    out = {f"patch_{d}": np.zeros((len(meshes), top), np.float32) for d in dims}
    patches = photos.vectors.shape[-2]  # 196 at 224 pixels, 784 at 448: the similarities grow with its square
    per_block = max(1, (VIEWS_PER_BLOCK // m1["views"]) * 196 ** 2 // patches ** 2)
    for start in tqdm(range(0, len(meshes), per_block), desc=f"{dataset}_{query_set}", unit="block"):
        block = slice(start, start + per_block)
        m = np.broadcast_to(meshes[block][..., None], views[block].shape)
        photo = torch.from_numpy(np.asarray(photos.vectors[photo_rows[block]])).to(device)
        photo_object = torch.from_numpy(np.asarray(photos.object[photo_rows[block]]) >= threshold).to(device)
        view = torch.from_numpy(np.asarray(renders.vectors[m, views[block]])).to(device)
        view_object = torch.from_numpy(np.asarray(renders.object[m, views[block]]) >= threshold).to(device)
        for d in dims:
            s = patch_scores(photo[..., :d], photo_object, view[..., :d], view_object)  # (B, top, V)
            out[f"patch_{d}"][block] = s.amax(-1).cpu().numpy()
    out["views"] = views.astype(np.int8)
    return out


def _choose_weight(sets: dict, key: str, weights: list[float], top: int, folds: np.ndarray | None = None) -> float:
    """The weight with the best pooled R@1 over the photos of `sets` (only the given folds, if any)."""
    best, best_hits = weights[0], -1.0
    for w in weights:
        hits = 0.0
        for s in sets.values():
            keep = np.isin(s["fold"], folds) if folds is not None else slice(None)
            ranks = reranked_ranks(s["rank"][keep], s["top"][keep], s["truth"][keep], s["top_score"][keep], s[key][keep], w, top)
            hits += (ranks <= 1).sum()
        if hits > best_hits:
            best, best_hits = w, hits
    return best


def report(cfg: dict, tag: str, datasets: list[str]) -> dict:
    """R@1 after method 1 for every set, with the weight chosen by leaving one fold out; save and return the report."""
    m1, (store, suffix) = cfg["rerank"]["method1"], store_of(cfg)
    top, weights = m1["top"], m1["weights"]
    sets = {}
    for dataset in datasets:
        for query_set in cand.QUERY_SETS[dataset]:
            lists = cand.read_candidates(tag, dataset, query_set)
            with np.load(method1_file(tag, dataset, query_set, m1["views"], suffix, top)) as data:
                sets[f"{dataset}_{query_set}"] = {**lists, **{k: data[k] for k in data.files}}
    folds = np.unique(np.concatenate([s["fold"] for s in sets.values()]))
    result = {"tag": tag, "store": store, "top": top, "views": m1["views"], "weights": weights, "dims": {}}
    for d in m1["dims"]:
        key = f"patch_{d}"
        held_out = {int(k): _choose_weight(sets, key, weights, top, folds[folds != k]) for k in folds}
        pooled_best = _choose_weight(sets, key, weights, top)
        rows = {}
        for name, s in sets.items():
            fold_weight = np.array([held_out[int(k)] for k in s["fold"]])
            ranks = np.empty_like(s["rank"])
            for k, w in held_out.items():
                keep = s["fold"] == k
                ranks[keep] = reranked_ranks(s["rank"][keep], s["top"][keep], s["truth"][keep], s["top_score"][keep], s[key][keep], w, top)
            curve = {str(w): float((reranked_ranks(s["rank"], s["top"], s["truth"], s["top_score"], s[key], w, top) <= 1).mean())
                     for w in weights}
            diff = paired_difference(s["rank"] <= 1, ranks <= 1, groups=s["truth"])
            rows[name] = {"photos": int(len(ranks)), "stage1_R@1": float((s["rank"] <= 1).mean()),
                          f"stage1_R@{top}": float((s["rank"] <= top).mean()), "method1_R@1": float((ranks <= 1).mean()),
                          "method1_R@5": float((ranks <= 5).mean()), "difference": diff,
                          "weights_used": sorted(set(fold_weight.tolist())), "R@1_by_weight": curve}
        result["dims"][str(d)] = {"held_out_weights": held_out, "pooled_best_weight": pooled_best, "sets": rows}
        print(f"\nmethod 1, {d} numbers per patch: weights by held-out fold {held_out}; best on all folds {pooled_best}")
        for name, r in rows.items():
            dd = r["difference"]
            print(f"  {name:24s} photos {r['photos']:6d}  stage 1 R@1 {100 * r['stage1_R@1']:5.1f}  -> method 1 {100 * r['method1_R@1']:5.1f}  "
                  f"({100 * dd['difference']:+.1f}, {100 * dd['low']:+.1f} to {100 * dd['high']:+.1f})  "
                  f"R@{top} limit {100 * r[f'stage1_R@{top}']:.1f}")
    target = cand.RERANK_DIR / tag / f"method1_v{m1['views']}{suffix}{top_mark(top)}_report.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {target.relative_to(paths.ROOT)}")
    return result


MESHES_PER_BLOCK = 24  # gallery stage: 24 meshes x 25 views x 196 x 196 similarities per call


def gallery_test(cfg: dict, tag: str, datasets: list[str], device: str = "cuda") -> dict:
    """The patch comparison as a stage 1: every mesh of the gallery, no shortlist.

    A random sample of `rerank.gallery.photos` val photos per dataset (SAM query images) is compared with
    every mesh of its own dataset, all views (no stage-1 view choice), for each number of dimensions in
    `rerank.gallery.dims`. Mesh score = the best view. The ranks are compared with the stage-1 ranks of
    the same photos (the out-of-fold lists). Saves and returns the report.
    """
    gcfg, (store, suffix) = cfg["rerank"]["gallery"], store_of(cfg)
    if suffix:
        raise ValueError("the gallery check runs on the store of the main image size only (its blocks hold the whole gallery "
                         "on the GPU); set rerank.patches.size to data.image_size")
    threshold = round(cfg["rerank"]["patches"]["object_share"] * 255)
    rng = np.random.default_rng(gcfg["seed"])
    result = {"tag": tag, "photos_per_dataset": gcfg["photos"], "dims": gcfg["dims"], "sets": {}}
    for dataset in datasets:
        lists = cand.read_candidates(tag, dataset, "queries_sam")
        pick = np.sort(rng.choice(len(lists["query_id"]), size=min(gcfg["photos"], len(lists["query_id"])), replace=False))
        photos = read_patches(patch_files(store, dataset, "queries_sam"))
        renders = read_patches(patch_files(store, dataset, "renders"))
        row = photos.row_index()
        d_max = max(gcfg["dims"])
        gallery = torch.from_numpy(np.asarray(renders.vectors[..., :d_max])).to(device)  # (M, V, Q, d)
        gallery_object = torch.from_numpy(np.asarray(renders.object) >= threshold).to(device)
        ranks = {d: np.zeros(len(pick), np.int64) for d in gcfg["dims"]}
        truth = lists["truth"][pick]
        for i, n in enumerate(tqdm(pick, desc=f"{dataset} gallery", unit="photo")):
            r = row[lists["query_id"][n]]
            photo = torch.from_numpy(np.asarray(photos.vectors[r:r + 1, :, :d_max])).to(device)
            photo_object = torch.from_numpy(np.asarray(photos.object[r:r + 1]) >= threshold).to(device)
            for d in gcfg["dims"]:
                mesh_score = torch.cat([
                    patch_scores(photo[..., :d], photo_object, gallery[None, start:start + MESHES_PER_BLOCK, ..., :d],
                                 gallery_object[None, start:start + MESHES_PER_BLOCK]).amax(-1)[0]
                    for start in range(0, len(gallery), MESHES_PER_BLOCK)]).cpu().numpy()
                ranks[d][i] = int((mesh_score >= mesh_score[truth[i]]).sum())
        stage1 = lists["rank"][pick]
        entry = {"photos": int(len(pick)), "stage1": {f"R@{k}": float((stage1 <= k).mean()) for k in (1, 5, 10, 50)}}
        for d in gcfg["dims"]:
            entry[f"patch_{d}"] = {f"R@{k}": float((ranks[d] <= k).mean()) for k in (1, 5, 10, 50)}
            entry[f"patch_{d}"]["difference_R@1"] = paired_difference(stage1 <= 1, ranks[d] <= 1, groups=truth)
        result["sets"][f"{dataset}_queries_sam"] = entry
        del gallery, gallery_object
        torch.cuda.empty_cache()
        print(f"{dataset}: {len(pick)} photos; stage 1 " + "  ".join(f"{k} {100 * v:.1f}" for k, v in entry["stage1"].items()))
        for d in gcfg["dims"]:
            e = entry[f"patch_{d}"]
            print(f"  patches only, {d} numbers: " + "  ".join(f"{k} {100 * e[k]:.1f}" for k in ("R@1", "R@5", "R@10", "R@50"))
                  + f"  (R@1 against stage 1 {100 * e['difference_R@1']['difference']:+.1f})")
    target = cand.RERANK_DIR / tag / f"gallery{suffix}_report.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {target.relative_to(paths.ROOT)}")
    return result


def dim_check(cfg: dict, tag: str, datasets: list[str], device: str = "cuda") -> dict:
    """How much the PCA reduction costs method 1: the full 768 numbers against the stored reduction.

    For a random sample of `rerank.dimcheck.photos` val photos per dataset (SAM query images), the
    backbone gives the raw patch tokens of the photo and of the best views of its first candidates
    (the method-1 setting); the patch scores come from the raw tokens (768) and from their projection
    (`rerank.dimcheck.dims`, the same projection as the store). R@1 after the re-ordering is compared
    at each weight of `rerank.method1.weights`. Saves and returns the report.
    """
    from src.models.backbones import FeatureExtractor, object_masks
    from src.models.patch_store import PatchProjection, projection_file
    from src.pipeline.extract_patches import set_images
    from PIL import Image

    m1, dcfg, backbone = cfg["rerank"]["method1"], cfg["rerank"]["dimcheck"], cfg["backbone"]["name"]
    object_share, background = cfg["rerank"]["patches"]["object_share"], tuple(cfg["data"]["background_rgb"])
    store, suffix = store_of(cfg)
    extractor = FeatureExtractor({**cfg["backbone"], "image_size": cfg["rerank"]["patches"]["size"]}, name=backbone, device=device)
    projection = PatchProjection.load(projection_file(store))
    rng = np.random.default_rng(dcfg["seed"])
    top, dims = m1["top"], [*dcfg["dims"], extractor.spec.dim]

    def tokens(files):
        images = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
        out = [extractor.extract_patches(images[i:i + extractor.batch_size], object_masks(images[i:i + extractor.batch_size], background))
               for i in range(0, len(images), extractor.batch_size)]
        return torch.cat([t for t, _ in out]), torch.cat([s for _, s in out]) >= object_share

    result = {"tag": tag, "store": store, "photos_per_dataset": dcfg["photos"], "sets": {}}
    for dataset in datasets:
        lists = cand.read_candidates(tag, dataset, "queries_sam")
        pick = np.sort(rng.choice(len(lists["query_id"]), size=min(dcfg["photos"], len(lists["query_id"])), replace=False))
        _, photo_labels, photo_files, _ = set_images(backbone, dataset, "queries_sam", suffix=suffix)
        _, _, render_list, views_per_mesh = set_images(backbone, dataset, "renders", suffix=suffix)
        photo_file = dict(zip(photo_labels["query_id"], photo_files))
        views = best_views(lists["view_score"][pick], top, m1["views"])  # (n, top, V)
        scores = {d: np.zeros((len(pick), top), np.float32) for d in dims}
        for start in tqdm(range(0, len(pick), 8), desc=f"{dataset} dims", unit="block"):
            idx = pick[start:start + 8]
            photo, photo_object = tokens([photo_file[q] for q in lists["query_id"][idx]])
            meshes = lists["top"][idx, :top]
            v = views[start:start + 8]
            files = [render_list[m * views_per_mesh + w] for m, vv in zip(meshes.ravel(), v.reshape(-1, v.shape[-1])) for w in vv]
            view, view_object = tokens(files)
            shape = (len(idx), top, v.shape[-1])
            view, view_object = view.view(*shape, *view.shape[1:]), view_object.view(*shape, -1)
            step = max(1, SIMILARITIES_PER_CALL // (top * v.shape[-1] * photo.shape[1] ** 2))  # 8 photos at 196 patches, 1 at 784
            for d in dims:
                p = photo if d == extractor.spec.dim else projection.project(photo, d)
                w = view if d == extractor.spec.dim else projection.project(view, d)
                block_scores = scores[d][start:start + len(idx)]  # a view: the rows of this block
                for j in range(0, len(idx), step):
                    part = slice(j, j + step)
                    block_scores[part] = patch_scores(p[part].half(), photo_object[part], w[part].half(), view_object[part]).amax(-1).cpu().numpy()
        entry = {"photos": int(len(pick)), "stage1_R@1": float((lists["rank"][pick] <= 1).mean()), "R@1_by_weight": {}}
        for d in dims:
            entry["R@1_by_weight"][str(d)] = {
                str(w): float((reranked_ranks(lists["rank"][pick], lists["top"][pick], lists["truth"][pick], lists["top_score"][pick],
                                             scores[d], w, top) <= 1).mean()) for w in m1["weights"]}
        result["sets"][f"{dataset}_queries_sam"] = entry
        print(f"{dataset}: {len(pick)} photos; stage 1 R@1 {100 * entry['stage1_R@1']:.1f}")
        for d in dims:
            curve = entry["R@1_by_weight"][str(d)]
            best = max(curve, key=curve.get)
            print(f"  {d:4d} numbers: best weight {best}: R@1 {100 * curve[best]:.1f};  " +
                  "  ".join(f"{w}: {100 * r:.1f}" for w, r in curve.items()))
    target = cand.RERANK_DIR / tag / f"dimcheck{suffix}_report.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {target.relative_to(paths.ROOT)}")
    return result


SIZECHECK_PHOTOS_PER_BLOCK = 8  # photos whose images are read together (8 x 51 images)


def size_check(cfg: dict, tag: str, datasets: list[str], device: str = "cuda") -> dict:
    """Method 1 with the patches of larger images: the same photos and candidates at each size of `rerank.sizecheck.sizes`.

    A larger image gives more patches of the same object (784 at 448 pixels against 196 at 224), so a thin
    detail (a leg, a seam) is no longer mixed into a patch with its neighbours. The backbone sees the images
    at each size (`FeatureExtractor.image_size`):
    - the main size (`data.image_size`): the images of the main folders, as in the patch store;
    - every other size: the images of `rerank.sizecheck.source_size` (for example `renders_448/`,
      `queries_448/`), shrunk in memory for a smaller size.
    Raw patch tokens, all 768 numbers (the PCA of the store was fitted on 224-pixel patches). The sample is a
    random choice of val photos per dataset (SAM query images); with the count and seed of dimcheck it is the
    same sample, so the main size repeats the 768-number row of `dim_check`.

    Per size: R@1 after the re-ordering for each weight, and R@1 with one weight for every source chosen by
    leaving one fold out (on the sample photos of the other folds, both datasets pooled), with the paired
    difference against stage 1 and against the main size. Saves and returns the report.
    """
    from src.models.backbones import FeatureExtractor, object_masks
    from src.pipeline.extract_patches import set_images
    from PIL import Image

    m1, scfg, backbone = cfg["rerank"]["method1"], cfg["rerank"]["sizecheck"], cfg["backbone"]["name"]
    object_share, background = cfg["rerank"]["patches"]["object_share"], tuple(cfg["data"]["background_rgb"])
    main, sizes, top, weights = cfg["data"]["image_size"], scfg["sizes"], m1["top"], m1["weights"]
    suffix_of = {s: "" if s == main else paths.size_suffix(scfg["source_size"], main) for s in sizes}
    extractor = FeatureExtractor(cfg["backbone"], name=backbone, device=device)
    rng = np.random.default_rng(scfg["seed"])

    def load(files):
        return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])

    def tokens(images, size):
        extractor.image_size = size
        b = extractor.batch_size
        out = [extractor.extract_patches(images[i:i + b], object_masks(images[i:i + b], background)) for i in range(0, len(images), b)]
        return torch.cat([t.half() for t, _ in out]), torch.cat([s for _, s in out]) >= object_share

    sets = {}
    for dataset in datasets:
        lists = cand.read_candidates(tag, dataset, "queries_sam")
        pick = np.sort(rng.choice(len(lists["query_id"]), size=min(scfg["photos"], len(lists["query_id"])), replace=False))
        files = {}
        for suffix in set(suffix_of.values()):
            _, labels, photo_files, _ = set_images(backbone, dataset, "queries_sam", suffix=suffix)
            _, _, render_list, views_per_mesh = set_images(backbone, dataset, "renders", suffix=suffix)
            files[suffix] = (dict(zip(labels["query_id"], photo_files)), render_list, views_per_mesh)
        views = best_views(lists["view_score"][pick], top, m1["views"])  # (n, top, V)
        scores = {s: np.zeros((len(pick), top), np.float32) for s in sizes}
        for start in tqdm(range(0, len(pick), SIZECHECK_PHOTOS_PER_BLOCK), desc=f"{dataset} sizes", unit="block"):
            idx = pick[start:start + SIZECHECK_PHOTOS_PER_BLOCK]
            meshes, v = lists["top"][idx, :top], views[start:start + SIZECHECK_PHOTOS_PER_BLOCK]
            shape = (len(idx), top, v.shape[-1])
            for suffix, (photo_file, render_list, views_per_mesh) in files.items():
                photo_images = load([photo_file[q] for q in lists["query_id"][idx]])
                view_images = load([render_list[m * views_per_mesh + w] for m, vv in zip(meshes.ravel(), v.reshape(-1, v.shape[-1])) for w in vv])
                for size in [s for s in sizes if suffix_of[s] == suffix]:
                    photo, photo_object = tokens(photo_images, size)
                    view, view_object = tokens(view_images, size)
                    view, view_object = view.view(*shape, *view.shape[1:]), view_object.view(*shape, -1)
                    step = max(1, SIMILARITIES_PER_CALL // (top * v.shape[-1] * photo.shape[1] ** 2))
                    block_scores = scores[size][start:start + len(idx)]  # a view: the rows of this block
                    for j in range(0, len(idx), step):
                        part = slice(j, j + step)
                        block_scores[part] = patch_scores(
                            photo[part], photo_object[part], view[part], view_object[part]).amax(-1).cpu().numpy()
        sets[f"{dataset}_queries_sam"] = {"rank": lists["rank"][pick], "top": lists["top"][pick], "truth": lists["truth"][pick],
                                          "top_score": lists["top_score"][pick], "fold": lists["fold"][pick],
                                          **{f"size_{s}": scores[s] for s in sizes}}
    extractor.image_size = cfg["backbone"]["image_size"]

    folds = np.unique(np.concatenate([s["fold"] for s in sets.values()]))
    result = {"tag": tag, "photos_per_dataset": scfg["photos"], "sizes": sizes, "source_size": scfg["source_size"],
              "top": top, "views": m1["views"], "weights": weights, "by_size": {}}
    held_out_ranks = {}
    for size in sizes:
        key = f"size_{size}"
        held_out = {int(k): _choose_weight(sets, key, weights, top, folds[folds != k]) for k in folds}
        rows = {}
        for name, s in sets.items():
            ranks = np.empty_like(s["rank"])
            for k, w in held_out.items():
                keep = s["fold"] == k
                ranks[keep] = reranked_ranks(s["rank"][keep], s["top"][keep], s["truth"][keep], s["top_score"][keep], s[key][keep], w, top)
            held_out_ranks[size, name] = ranks
            rows[name] = {"photos": int(len(ranks)), "stage1_R@1": float((s["rank"] <= 1).mean()),
                          "R@1": float((ranks <= 1).mean()),
                          "difference_to_stage1": paired_difference(s["rank"] <= 1, ranks <= 1, groups=s["truth"]),
                          "difference_to_main_size": paired_difference(held_out_ranks[main, name] <= 1, ranks <= 1, groups=s["truth"])
                          if (main, name) in held_out_ranks else None,
                          "R@1_by_weight": {str(w): float((reranked_ranks(s["rank"], s["top"], s["truth"], s["top_score"], s[key], w, top) <= 1).mean())
                                            for w in weights}}
        result["by_size"][str(size)] = {"held_out_weights": held_out, "sets": rows}
        print(f"\n{size} pixels ({(size // extractor.spec.patch_size) ** 2} patches): weights by held-out fold {held_out}")
        for name, r in rows.items():
            d, dm = r["difference_to_stage1"], r["difference_to_main_size"]
            print(f"  {name:20s} stage 1 R@1 {100 * r['stage1_R@1']:5.1f} -> {100 * r['R@1']:5.1f} "
                  f"({100 * d['difference']:+.1f}, {100 * d['low']:+.1f} to {100 * d['high']:+.1f})"
                  + (f"; against {main}: {100 * dm['difference']:+.1f} ({100 * dm['low']:+.1f} to {100 * dm['high']:+.1f})" if dm else ""))
    target = cand.RERANK_DIR / tag / "sizecheck_report.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {target.relative_to(paths.ROOT)}")
    return result


def run(cfg: dict, stages: list[str], split: str, tag: str, datasets: list[str], folds: int, device: str = "cuda") -> None:
    if "candidates" in stages:
        cand.build(cfg, split, tag, datasets, folds, device)
    if "method1" in stages:
        for dataset in datasets:
            for query_set in cand.QUERY_SETS[dataset]:
                out = method1_scores(cfg, tag, dataset, query_set, device)
                m1 = cfg["rerank"]["method1"]
                np.savez(method1_file(tag, dataset, query_set, m1["views"], store_of(cfg)[1], m1["top"]), **out)
    if "report" in stages:
        report(cfg, tag, datasets)
    if "gallery" in stages:
        gallery_test(cfg, tag, datasets, device)
    if "dimcheck" in stages:
        dim_check(cfg, tag, datasets, device)
    if "sizecheck" in stages:
        size_check(cfg, tag, datasets, device)
