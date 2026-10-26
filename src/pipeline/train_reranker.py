"""Re-ranking method 2: train and measure the pair scorer (`src/models/reranker.py`) by cross-validation.

The pair scorer re-orders the first `rerank.method2.top` meshes of each photo, as method 1 does. It is
trained and measured on the out-of-fold lists of a stage-1 cross-validation (`src/retrieval/candidates.py`):

- **Re-ranker k** trains on the photos of the train part of fold k (their lists come from the other
  fold models, so stage 1 did not train on them) and is measured on the val photos of fold k.
- **Training lists, no held-out mesh (rule R2):** the stored list of a photo has its first 50 meshes;
  only the meshes of the fold's training photos are kept (as in the stage-1 training gallery: no
  held-out, test, or photo-less mesh, not even as a negative). The first `top` of these, plus the
  correct mesh as the 11th (or the next kept mesh, if the correct mesh is already among them), so
  every list has `top` + 1 meshes. A photo whose correct mesh is not in its first 50 is left out.
- **Measuring lists:** the first `top` meshes of the stored list, as in use (no filter, no added
  correct mesh); a correct mesh outside them keeps its stage-1 rank (`reranked_ranks`).
- **Pix3D:** each photo has a reference image and a SAM image; in training one of the two is taken at
  random for each use (as in the heads training); both sets are measured.
- **Method 1** (the patch score, `patch_scores`) is computed here from the full patches, because the
  training lists contain meshes that the stored method-1 files do not have. On the measuring lists it
  equals the stored method-1 score (checked in each evaluation).

Loss: cross-entropy over the `top` + 1 meshes of a training photo, of final score / tau
(`final_scores`; the variant decides whether method 1 is part of the base score).

Outputs (a mini run writes into `test_outputs/train_reranker_mini/` instead):
    runs/<run_name>/config.yaml, train_log.csv, val_ranks.npz, metrics.json     one run per fold
    weights/checkpoints/<run_name>/reranker_last.pt                              the pair scorer after the last epoch
    data/derived/rerank/<tag>/method2_<variant><size suffix>_report.json        pooled over the folds
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import paths, runs
from src.data.catalog import read_table
from src.data.workers import loader_worker_init
from src.models.patch_store import patch_files, read_patches
from src.models.reranker import VARIANTS, PairScorer, final_scores, select_tokens
from src.pipeline.rerank import best_views, method1_file, store_of
from src.retrieval import candidates as cand
from src.retrieval.metrics import paired_difference
from src.retrieval.patch_rerank import patch_scores, reranked_ranks
from src.train.trainer import learning_rate_factor

SPLIT = "unseen_meshes"
RANK_PARTS = ("query_id", "truth", "stage1", "method1", "method2")  # the arrays of each set in val_ranks.npz


def output_dirs(mini: bool) -> dict[str, Path]:
    if mini:
        base = paths.TEST_OUTPUTS_DIR / "train_reranker_mini"
        return {"runs": base / "runs", "checkpoints": base / "checkpoints", "reports": base / "rerank"}
    return {"runs": paths.RUNS_DIR, "checkpoints": paths.CHECKPOINTS_DIR, "reports": cand.RERANK_DIR}


# ---------------------------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------------------------


class QuerySet:
    """One query set: its stored lists, its photo patches, and the render patches of its dataset."""

    def __init__(self, tag: str, dataset: str, query_set: str, store: str, renders):
        self.name, self.dataset = f"{dataset}_{query_set}", dataset
        self.lists = cand.read_candidates(tag, dataset, query_set)
        self.photos = read_patches(patch_files(store, dataset, query_set))
        self.renders = renders
        if not np.array_equal(renders.ids, self.lists["gallery"]):
            raise ValueError(f"{self.name}: the render rows of the patch store and the gallery of the lists differ")
        row = self.photos.row_index()
        self.photo_rows = np.array([row[q] for q in self.lists["query_id"]])


def training_list(top_meshes: np.ndarray, allowed: np.ndarray, truth: int, top: int) -> tuple[np.ndarray, int] | None:
    """Positions (in the stored list) of the `top` + 1 meshes of one training photo, and the index of the correct mesh.

    Args:
        top_meshes: the stored list (gallery indices, best first). allowed: bool over the gallery (rule R2).

    Returns None if the correct mesh is not in the stored list or too few meshes are allowed.
    """
    kept = np.flatnonzero(allowed[top_meshes])
    where = np.flatnonzero(top_meshes == truth)
    if not len(where) or not allowed[truth]:
        return None
    truth_position = where[0]
    first = kept[:top]
    chosen = kept[:top + 1] if truth_position in first else np.append(first, truth_position)
    if len(chosen) < top + 1:
        return None
    return chosen, int(np.flatnonzero(chosen == truth_position)[0])


def allowed_meshes(dataset: str, fold: int, gallery: np.ndarray) -> tuple[np.ndarray, set[str]]:
    """(bool over the gallery: the meshes of the training photos of fold `fold`, as the stage-1 training gallery;
    the query ids of these training photos)."""
    train = set(json.loads(paths.split_file(dataset, f"{SPLIT}_fold{fold}").read_text())["parts"]["train"])
    meshes = {q["mesh_id"] for q in read_table(paths.QUERIES_CSV) if q["dataset"] == dataset and q["query_id"] in train}
    return np.isin(gallery, list(meshes)), train


class PairData(Dataset):
    """Items for the pair scorer. Each item is a list of choices (set, list row, stored positions, target);
    one choice is taken at random (Pix3D: the reference or the SAM image of the same photo)."""

    def __init__(self, sets: dict[str, QuerySet], items: list[list[tuple]], views: int, seed: int):
        self.sets, self.items, self.views = sets, items, views
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        choices = self.items[index]
        name, i, positions, target = choices[self.rng.integers(len(choices))] if len(choices) > 1 else choices[0]
        s = self.sets[name]
        meshes = s.lists["top"][i, positions]
        views = best_views(s.lists["view_score"][i][None, positions], len(positions), self.views)[0]  # (C, V)
        m = np.broadcast_to(meshes[:, None], views.shape)
        r = s.photo_rows[i]
        return {"photo": np.asarray(s.photos.vectors[r]), "photo_object": np.asarray(s.photos.object[r]),
                "view": np.asarray(s.renders.vectors[m, views]), "view_object": np.asarray(s.renders.object[m, views]),
                "stage1": s.lists["top_score"][i, positions].astype(np.float32), "target": target}


def _collate(batch: list[dict]) -> dict:
    return {key: torch.from_numpy(np.stack([b[key] for b in batch])) for key in batch[0]}


# ---------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def method1_batch(photo, photo_object, view, view_object) -> torch.Tensor:
    """Method-1 score of every listed mesh (the best of its views), one photo at a time: (B, C)."""
    return torch.cat([patch_scores(photo[b:b + 1], photo_object[b:b + 1], view[b:b + 1], view_object[b:b + 1]).amax(-1)
                      for b in range(len(photo))])


def corrections(model: PairScorer, batch: dict, on_photo, on_view, tokens: int, generator) -> torch.Tensor:
    """The correction r of every listed mesh: the best of its views' r_v. (B, C)."""
    photo, view = batch["photo"], batch["view"]
    b, c, v, p, d = view.shape
    pt, pp, pm = select_tokens(photo, on_photo, tokens, generator)
    vt, vp, vm = select_tokens(view.reshape(-1, p, d), on_view.reshape(-1, p), tokens, generator)
    pairs = c * v
    r = model(pt.repeat_interleave(pairs, 0), pp.repeat_interleave(pairs, 0), pm.repeat_interleave(pairs, 0), vt, vp, vm)
    return r.float().view(b, c, v).amax(-1)


def _to_device(batch: dict, device: str, threshold: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return batch, batch["photo_object"] >= threshold, batch["view_object"] >= threshold


@torch.no_grad()
def evaluate(model: PairScorer, sets: dict[str, QuerySet], rows: dict[str, np.ndarray], m2: dict, threshold: int,
             device: str) -> dict[str, dict[str, np.ndarray]]:
    """Method 1 and method 2 scores of the first `top` meshes of the given rows of each set."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(m2["seed"])  # the same token choice in every evaluation
    out = {}
    for name, idx in rows.items():
        top = m2["top"]
        items = [[(name, i, np.arange(top), -1)] for i in idx]
        loader = DataLoader(PairData(sets, items, m2["views"], 0), batch_size=m2["eval_photos"], shuffle=False,
                            num_workers=m2["loader_workers"], collate_fn=_collate, worker_init_fn=loader_worker_init, pin_memory=True)
        m1, r = [], []
        for batch in loader:
            batch, on_photo, on_view = _to_device(batch, device, threshold)
            m1.append(method1_batch(batch["photo"], on_photo, batch["view"], on_view).cpu())
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                r.append(corrections(model, batch, on_photo, on_view, m2["tokens"], generator).cpu())
        out[name] = {"method1": torch.cat(m1).numpy(), "correction": torch.cat(r).numpy()}
    model.train()
    return out


def ranks_of(s: QuerySet, idx: np.ndarray, scores: dict, m2: dict) -> tuple[np.ndarray, np.ndarray]:
    """(method-1 ranks, method-2 ranks) of the correct mesh for the given rows."""
    L, top, w = s.lists, m2["top"], m2["method1_weight"]
    args = (L["rank"][idx], L["top"][idx], L["truth"][idx], L["top_score"][idx])
    method1 = reranked_ranks(*args, scores["method1"], w, top)
    extra = w * scores["method1"] + scores["correction"] if m2["variant"] == "residual_method1" else scores["correction"]
    return method1, reranked_ranks(*args, extra, 1.0, top)


# ---------------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------------


def _fold_items(sets: dict[str, QuerySet], fold: int, top: int, mini: dict | None, rng) -> tuple[list, dict, dict]:
    """Training items (rule R2), val rows of each set, and counts, for re-ranker `fold`."""
    items, val_rows, counts = {}, {}, {"left_out": {}}
    for name, s in sets.items():
        allowed, train = allowed_meshes(s.dataset, fold, s.lists["gallery"])
        val_rows[name] = np.flatnonzero(s.lists["fold"] == fold)
        rows = np.flatnonzero(np.isin(s.lists["query_id"], list(train)))
        if (s.lists["fold"][rows] == fold).any():
            raise ValueError(f"{name}: a training photo of fold {fold} has a list of the fold-{fold} model")
        left = 0
        for i in rows:
            chosen = training_list(s.lists["top"][i], allowed, s.lists["truth"][i], top)
            if chosen is None:
                left += 1
                continue
            items.setdefault(s.lists["query_id"][i], []).append((name, i, chosen[0], chosen[1]))
        counts["left_out"][name] = left
        counts[f"allowed_meshes_{s.dataset}"] = int(allowed.sum())
    items = list(items.values())
    if mini:
        items = [items[i] for i in rng.choice(len(items), size=min(mini["train_photos"], len(items)), replace=False)]
        val_rows = {n: np.sort(rng.choice(r, size=min(mini["val_photos"], len(r)), replace=False)) for n, r in val_rows.items()}
    counts["training_photos"] = len(items)
    counts["val_photos"] = {n: int(len(r)) for n, r in val_rows.items()}
    return items, val_rows, counts


def train_fold(cfg: dict, sets: dict[str, QuerySet], fold: int, tag: str, out: dict, mini: bool, device: str) -> Path:
    m2 = cfg["rerank"]["method2"]
    store, _ = store_of(cfg)
    threshold = round(cfg["rerank"]["patches"]["object_share"] * 255)
    rng = np.random.default_rng(m2["seed"] + fold)
    torch.manual_seed(m2["seed"] + fold)
    items, val_rows, counts = _fold_items(sets, fold, m2["top"], m2["mini"] if mini else None, rng)
    epochs = 1 if mini else m2["epochs"]
    run = runs.create_run(cfg, "pix3d_abo", store, f"{SPLIT}_fold{fold}_m2{m2['variant']}_{tag}", out["runs"], out["checkpoints"])
    model = PairScorer(sets[next(iter(sets))].photos.vectors.shape[-1], m2["width"], m2["heads"], m2["blocks"], m2["ffn"]).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": m2["weight_decay"]},
                                   {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}], lr=m2["lr"])
    loader = DataLoader(PairData(sets, items, m2["views"], m2["seed"] + fold), batch_size=m2["batch_photos"], shuffle=True,
                        drop_last=True, num_workers=m2["loader_workers"], collate_fn=_collate,
                        worker_init_fn=loader_worker_init, pin_memory=True, persistent_workers=True)
    total = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, m2["warmup_epochs"] * len(loader), total))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    generator = torch.Generator(device=device).manual_seed(m2["seed"] + fold)
    weights = sum(p.numel() for p in model.parameters())
    print(f"run {run['run_name']}: {counts['training_photos']} training photos, val {counts['val_photos']}, "
          f"left out (correct mesh not in the first 50 / too few allowed) {counts['left_out']}; {weights:,} weights; "
          f"{len(loader)} steps per epoch, {epochs} epochs", flush=True)

    log_rows, epoch_ranks = [], {name: [] for name in sets}
    stage1_ranks = {name: sets[name].lists["rank"][r] for name, r in val_rows.items()}
    method1_ranks, epoch_numbers = {}, []
    for epoch in range(1, epochs + 1):
        start, losses, hits = time.perf_counter(), [], []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader):
            batch, on_photo, on_view = _to_device(batch, device, threshold)
            m1 = method1_batch(batch["photo"], on_photo, batch["view"], on_view)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                r = corrections(model, batch, on_photo, on_view, m2["tokens"], generator)
            scores = final_scores(m2["variant"], batch["stage1"], m1, r, m2["method1_weight"])
            loss = torch.nn.functional.cross_entropy(scores / m2["tau"], batch["target"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            hits.append((scores.argmax(1) == batch["target"]).float().mean().item())
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-200:]):.4f} "
                      f"batch R@1 {100 * np.mean(hits[-200:]):.1f} {(step + 1) * m2['batch_photos'] / (time.perf_counter() - start):.1f} photos/s", flush=True)
        train_seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        if m2["eval_epochs"] == "last" and epoch < epochs:
            print(f"epoch {epoch}: loss {np.mean(losses):.4f}, {len(loader) * m2['batch_photos'] / train_seconds:.1f} photos/s, "
                  f"train {train_seconds:.0f} s (no evaluation: rerank.method2.eval_epochs = last)", flush=True)
            continue
        scores = evaluate(model, sets, val_rows, m2, threshold, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "batch_R@1": float(np.mean(hits)), "lr_end": scheduler.get_last_lr()[0],
               "train_seconds": round(train_seconds, 1), "eval_seconds": round(time.perf_counter() - start - train_seconds, 1),
               "photos_per_second": round(len(loader) * m2["batch_photos"] / train_seconds, 2), "peak_gpu_mb": peak}
        for name, idx in val_rows.items():
            m1_ranks, m2_ranks = ranks_of(sets[name], idx, scores[name], m2)
            method1_ranks[name] = m1_ranks
            epoch_ranks[name].append(m2_ranks)
            row[f"{name}_stage1_R@1"] = float((stage1_ranks[name] <= 1).mean())
            row[f"{name}_method1_R@1"] = float((m1_ranks <= 1).mean())
            row[f"{name}_method2_R@1"] = float((m2_ranks <= 1).mean())
            with np.load(method1_file(tag, sets[name].dataset, name.split("_", 1)[1], m2["views"], store_of(cfg)[1])) as f:
                stored = f["patch_128"][idx, :m2["top"]]  # method 1 as stored by 13_rerank.py: must be the same
            row[f"{name}_method1_check"] = float(np.abs(stored - scores[name]["method1"]).max())
        log_rows.append(row)
        epoch_numbers.append(epoch)
        print(f"epoch {epoch}: loss {row['loss']:.4f}, {row['photos_per_second']} photos/s, peak {peak:.0f} MB, train {row['train_seconds']} s, "
              f"eval {row['eval_seconds']} s; val R@1 (stage 1 / method 1 / method 2): " +
              "; ".join(f"{n} {100 * row[f'{n}_stage1_R@1']:.1f} / {100 * row[f'{n}_method1_R@1']:.1f} / {100 * row[f'{n}_method2_R@1']:.1f}"
                        for n in val_rows), flush=True)

    with open(run["run_dir"] / "train_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        writer.writeheader(); writer.writerows(log_rows)
    arrays = {}
    for name, idx in val_rows.items():
        arrays[f"{name}_query_id"] = sets[name].lists["query_id"][idx]
        arrays[f"{name}_truth"] = sets[name].lists["truth"][idx]
        arrays[f"{name}_stage1"] = stage1_ranks[name]
        arrays[f"{name}_method1"] = method1_ranks[name]
        arrays[f"{name}_method2"] = np.stack(epoch_ranks[name])  # (evaluated epochs, photos)
    np.savez(run["run_dir"] / "val_ranks.npz", **arrays)
    commit, dirty = runs.git_state()
    meta = {"run_name": run["run_name"], "fold": fold, "evaluated_epochs": epoch_numbers, "tag": tag, "store": store, "variant": m2["variant"],
            "method2": m2, "weights": weights, "git_commit": commit, "git_dirty": dirty, "counts": counts}
    torch.save({"model": model.state_dict(), "meta": meta}, run["checkpoint_dir"] / "reranker_last.pt")
    (run["run_dir"] / "metrics.json").write_text(json.dumps({**meta, "epochs": log_rows}, indent=2, default=str) + "\n")
    return run["run_dir"]


def pool(run_dirs: list[Path], m2: dict) -> dict:
    """Pooled R@1 of the folds (last epoch, and per epoch), with paired differences over meshes."""
    data: dict[str, dict[str, list]] = {}
    for run_dir in run_dirs:
        with np.load(run_dir / "val_ranks.npz") as f:
            for key in f.files:
                part = next(p for p in RANK_PARTS if key.endswith("_" + p))
                data.setdefault(key[:-len(part) - 1], {}).setdefault(part, []).append(f[key])
    result = {}
    for name, parts in data.items():
        stage1, method1, truth = (np.concatenate(parts[k]) for k in ("stage1", "method1", "truth"))
        method2 = np.concatenate(parts["method2"], axis=1)  # (epochs, photos)
        last = method2[-1]
        result[name] = {"photos": int(len(truth)), "stage1_R@1": float((stage1 <= 1).mean()),
                        "method1_R@1": float((method1 <= 1).mean()), "method2_R@1": float((last <= 1).mean()),
                        "method2_R@5": float((last <= 5).mean()),
                        "method2_R@1_by_epoch": [float((e <= 1).mean()) for e in method2],
                        "against_method1": paired_difference(method1 <= 1, last <= 1, groups=truth),
                        "against_stage1": paired_difference(stage1 <= 1, last <= 1, groups=truth)}
    return result


def run(cfg: dict, tag: str, folds: list[int], datasets: list[str], mini: bool = False, device: str = "cuda") -> dict:
    m2 = cfg["rerank"]["method2"]
    if m2["variant"] not in VARIANTS:
        raise ValueError(f"rerank.method2.variant must be one of {VARIANTS}")
    store, suffix = store_of(cfg)
    out = output_dirs(mini)
    renders = {d: read_patches(patch_files(store, d, "renders")) for d in datasets}
    sets = {f"{d}_{q}": QuerySet(tag, d, q, store, renders[d]) for d in datasets for q in cand.QUERY_SETS[d]}
    folds = folds[:1] if mini else folds
    run_dirs = [train_fold(cfg, sets, k, tag, out, mini, device) for k in folds]
    report = {"tag": tag, "store": store, "variant": m2["variant"], "folds": folds, "method2": m2, "runs": [d.name for d in run_dirs],
              "sets": pool(run_dirs, m2)}
    target = out["reports"] / tag / f"method2_{m2['variant']}{suffix}_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nmethod 2 ({m2['variant']}), pooled over folds {folds}:")
    for name, r in report["sets"].items():
        d = r["against_method1"]
        print(f"  {name:24s} photos {r['photos']:6d}  stage 1 {100 * r['stage1_R@1']:5.1f}  method 1 {100 * r['method1_R@1']:5.1f}  "
              f"method 2 {100 * r['method2_R@1']:5.1f}  (against method 1 {100 * d['difference']:+.1f}, {100 * d['low']:+.1f} to {100 * d['high']:+.1f})")
    print(f"-> {target}")
    return report
