"""The training step: train the three embedding heads on one split, and record the run.

Steps:

1. the training photos of the split's train part (`src/data/datasets.py`), the training gallery
   (the meshes of these photos), and the full gallery and the val photos of each dataset (stored
   features, `src/train/trainer.py`);
2. the frozen backbone for the changed training images, the heads, AdamW, and the learning-rate
   schedule (warm-up, then cosine);
3. for each epoch: one pass over the training photos, then the validation; the heads of the epoch
   with the best selection value (mean val R@1 over the val query sets) are saved;
4. the run records.

Outputs (a mini run writes into `test_outputs/train_mini/` instead):
    runs/<run_name>/config.yaml               the settings of the run
    runs/<run_name>/train_log.csv             one row per epoch: losses, learning rate, speed, val metrics
    runs/<run_name>/metrics.json              the best epoch, its val metrics, the zero-shot val R@1, counts
    weights/checkpoints/<run_name>/embedding_heads_best.pt   the heads of the best epoch (self-describing)
    weights/checkpoints/<run_name>/embedding_heads_last.pt   the heads after the last epoch
    runs/registry.csv                         one row per val query set (the best epoch)

A mini run uses 5 % of the training photos (`train.mini_fraction`) and 1 epoch; it checks that the
step works from start to end.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import paths, runs
from src.data.augment import QueryAugment
from src.data.catalog import read_table
from src.data.datasets import TrainingQueries, loader_worker_init, training_photos
from src.models.backbones import FeatureExtractor
from src.models.embedding_heads import EmbeddingHeads, parameter_counts
from src.pipeline.extract_features import output_root as features_output_root
from src.train import trainer

HEAD_NAMES = runs.HEAD_KEYS


def output_dirs(mini: bool) -> dict[str, Path]:
    """Return the run folder root, the checkpoint root, and the registry (test copies for a mini run)."""
    if mini:
        base = paths.TEST_OUTPUTS_DIR / "train_mini"
        return {"runs": base / "runs", "checkpoints": base / "checkpoints", "registry": base / "registry.csv"}
    return {"runs": paths.RUNS_DIR, "checkpoints": paths.CHECKPOINTS_DIR, "registry": paths.REGISTRY_CSV}


def _flatten(prefix: str, results: dict[str, dict]) -> dict:
    return {f"{prefix}{name}_{key}": round(value, 4) for name, entry in results.items() for key, value in entry.items()
            if key in ("R@1", "R@5", "cat@1", "trained_R@1")}


def run(cfg: dict, split: str = "seen_meshes", datasets: list[str] | None = None, backbone: str | None = None,
        tag: str | None = None, mini: bool = False, device: str = "cuda") -> dict:
    """Train the heads on one split; return the content of metrics.json.

    Args:
        cfg: The merged settings; the step reads `train`, `backbone`, `data`, and `render`.
        split: The split, for example "seen_meshes".
        datasets: The datasets of the training (None = `data.datasets`).
        backbone: The backbone of the feature files (None = `backbone.name`).
        tag: Extra word for the run name.
        mini: True = 5 % of the training photos, 1 epoch, outputs in test_outputs/train_mini/.
        device: "cuda" or "cpu".
    """
    train_cfg, data_cfg = cfg["train"], cfg["data"]
    datasets = datasets or list(data_cfg["datasets"])
    backbone = backbone or cfg["backbone"]["name"]
    recipe = cfg["backbone"]["feature_recipe"]
    epochs = 1 if mini else train_cfg["epochs"]
    out = output_dirs(mini)
    runs.set_seed(train_cfg["seed"])
    generator = torch.Generator().manual_seed(train_cfg["seed"])  # view dropout and the order of the photos

    # 1. Photos, galleries, val sets.
    category_index = {c: i for i, c in enumerate(sorted({q["category"] for q in read_table(paths.QUERIES_CSV)}))}
    photos, photo_report = training_photos(datasets, split, "train", train_cfg["query_images"])
    if mini:
        keep = np.random.default_rng(train_cfg["seed"]).choice(len(photos), size=max(1, int(len(photos) * train_cfg["mini_fraction"])), replace=False)
        photos = [photos[i] for i in sorted(keep)]
    features_root = features_output_root(False)  # the real feature files (the mini run reads them; it writes nothing there)
    galleries, feature_files = trainer.load_galleries(datasets, backbone, recipe, category_index, features_root, device)
    gallery = trainer.training_gallery(galleries, {p.mesh_id for p in photos})
    mesh_index = {m: i for i, m in enumerate(gallery.mesh_ids)}
    val_sets = trainer.load_val_sets(datasets, split, backbone, recipe, galleries, category_index,
                                     cfg["backbone"]["zero_shot"], features_root, device)

    # 2. Backbone, heads, data loader, optimizer, schedule.
    extractor = FeatureExtractor(cfg["backbone"], name=backbone, device=device)
    heads = EmbeddingHeads.from_config(train_cfg["heads"], extractor.dim).to(device)
    augment = QueryAugment(train_cfg["augment"], data_cfg["background_rgb"], data_cfg["image_size"])
    dataset = TrainingQueries(photos, augment, mesh_index, category_index, train_cfg["seed"])
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=len(photos) > train_cfg["batch_size"],
                        num_workers=cfg["backbone"]["loader_workers"], worker_init_fn=loader_worker_init,
                        persistent_workers=True, generator=generator)
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.AdamW(trainer.parameter_groups(heads, opt_cfg["weight_decay"]), lr=opt_cfg["lr"])
    total_steps = epochs * len(loader)
    warmup_steps = min(total_steps, int(train_cfg["schedule"]["warmup_epochs"] * len(loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: trainer.learning_rate_factor(step, warmup_steps, total_steps))

    # The run: name, folders, settings.
    label = "_".join(datasets)
    run = runs.create_run(cfg, label, backbone, "_".join(p for p in (split, tag) if p), runs_dir=out["runs"], checkpoints_dir=out["checkpoints"])
    print(f"run {run['run_name']}: {len(photos)} training photos, training gallery {len(gallery.mesh_ids)} meshes, "
          f"{len(loader)} steps per epoch, {epochs} epochs; heads {parameter_counts(heads)['total']:,} parameters", flush=True)
    print("zero-shot val R@1: " + ", ".join(f"{v.name} {100 * v.zero_shot_r1:.1f}" for v in val_sets), flush=True)

    # 3. Epochs.
    log_rows, best = [], {"value": -1.0}
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_values = trainer.train_epoch(heads, extractor, loader, optimizer, scheduler, gallery, train_cfg,
                                           data_cfg["background_rgb"], generator, device, desc=f"epoch {epoch}/{epochs}")
        results = trainer.validate(heads, val_sets, galleries, train_cfg["scoring"]["alpha"], train_cfg["scoring"]["mix_weight"])
        value = trainer.selection_value(results)
        row = {"epoch": epoch, "lr": round(scheduler.get_last_lr()[0], 7),
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in train_values.items()},
               "selection_R@1": round(value, 4), **_flatten("val_", results)}
        log_rows.append(row)
        heads_state = {name: getattr(heads, name).state_dict() for name in HEAD_NAMES}
        if value > best["value"]:
            best = {"value": value, "epoch": epoch, "results": results}
            meta = runs.checkpoint_metadata(run["run_name"], cfg, backbone, feature_files, train_cfg["heads"]["embedding_dim"], epoch, value)
            runs.save_checkpoint(run["checkpoint_dir"] / "embedding_heads_best.pt", heads_state, meta)
        print(f"epoch {epoch:3d}/{epochs}  loss {row['loss']:.4f} (instance {row['instance']:.4f}, category {row['category']:.4f})  "
              f"batch R@1 {100 * row['batch_R@1']:5.1f}  val R@1 " + ", ".join(f"{n} {100 * r['R@1']:.1f}" for n, r in results.items())
              + " (trained alone " + ", ".join(f"{100 * r['trained_R@1']:.1f}" for r in results.values()) + ")"
              + f"  | {row['images_per_second']} images/s, loader wait {100 * row['loader_wait_share']:.0f} %"
              + ("  *best" if best["epoch"] == epoch else ""), flush=True)  # flush: show the line at once, also through a pipe
        with open(run["run_dir"] / "train_log.csv", "w", newline="") as f:  # rewritten each epoch, so a stopped run keeps its log
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            writer.writeheader()
            writer.writerows(log_rows)
    meta = runs.checkpoint_metadata(run["run_name"], cfg, backbone, feature_files, train_cfg["heads"]["embedding_dim"], epochs, value)
    runs.save_checkpoint(run["checkpoint_dir"] / "embedding_heads_last.pt", heads_state, meta)

    # 4. Records.
    commit, dirty = runs.git_state()
    record = {
        "run_name": run["run_name"],
        "split": split,
        "datasets": datasets,
        "backbone": backbone,
        "recipe": recipe,
        "mix_weight": train_cfg["scoring"]["mix_weight"],
        "temperature": train_cfg["loss"]["temperature"],
        "mini": mini,
        "training_photos": len(photos),
        "training_images": photo_report["images"],
        "training_gallery_meshes": len(gallery.mesh_ids),
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_selection_R@1": best["value"],
        "best_val": best["results"],
        "parameters": parameter_counts(heads),
        "minutes": round((time.perf_counter() - start) / 60, 1),
        "git_commit": commit,
        "git_dirty": dirty,
    }
    (run["run_dir"] / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    checkpoint = run["checkpoint_dir"] / "embedding_heads_best.pt"
    for name, result in best["results"].items():
        set_dataset, query_set = name.split("_", 1)  # "pix3d_queries_reference" -> "pix3d", "queries_reference"
        runs.append_registry({
            "run_name": run["run_name"], "date": time.strftime("%Y-%m-%d"), "git_commit": commit[:7], "git_dirty": dirty,
            "mode": "trained", "datasets": label, "split": split, "photos": "val", "photo_count": result["photos"],
            "query_set": query_set, "backbone": backbone, "recipe": recipe, "scoring": "combined",
            "alpha": train_cfg["scoring"]["alpha"], "mix_weight": train_cfg["scoring"]["mix_weight"], "render_config": cfg["render"]["rig"], "lighting": cfg["render"]["lighting"],
            "temperature": train_cfg["loss"]["temperature"], "category_weight": train_cfg["loss"]["category_weight"],
            "epochs": epochs, "best_epoch": best["epoch"], "val_R@1": round(100 * best["value"], 2),
            "R@1": round(100 * result["R@1"], 2), "R@5": round(100 * result["R@5"], 2), "R@10": round(100 * result["R@10"], 2),
            "cat@1": round(100 * result["cat@1"], 2),
            "checkpoint_path": str(checkpoint.relative_to(paths.ROOT)),
            "notes": f"{set_dataset}; score mix {train_cfg['scoring']['mix_weight']} trained + zero-shot; trained alone {100 * result['trained_R@1']:.1f}; "
                     f"zero-shot val R@1 {100 * result['zero_shot_R@1']:.1f}" + ("; mini" if mini else ""),
        }, registry=out["registry"])
    print(f"best epoch {best['epoch']} (mean val R@1 {100 * best['value']:.1f}); {record['minutes']} min -> {run['run_dir'].relative_to(paths.ROOT)}", flush=True)
    return record
