"""Reader for the Amazon Berkeley Objects (ABO) dataset: metadata, class selection, identifiers.

ABO (Collins et al., CVPR 2022) is a product catalog with 3D meshes and catalog photos. The data
is in a public bucket (anonymous access) and has the licence CC BY 4.0 (credit: Amazon.com).
Three metadata files describe it:

- `3dmodels.csv.gz`: one row per 3D model: model id, `.glb` path in the bucket, vertex and face
  counts, and the size of the model in metres (extent_x, extent_y, extent_z).
- `listings_0.json.gz` ... `listings_f.json.gz`: one JSON line per product listing. A listing
  with a "3dmodel_id" links a 3D model to its product type (for example "CHAIR"), its category
  path (for example "/Furniture/Living Room Furniture/Chairs"), and its photos ("main_image_id",
  "other_image_id"). One model has 1 to 3 listings (other marketplaces or languages).
- `images.csv.gz`: one row per photo: image id, height, width, and path in the bucket.

Class selection: the class rules in `configs/data.yaml` (`data.abo.class_rules`) turn the
product type of a model into one of our classes. A rule can also require that the last part of
the category path ("leaf") matches a regular expression; this splits mixed product types (for
example HOME_FURNITURE_AND_DECOR contains TV stands and carts). The first matching rule wins.
Models that no rule matches are not used.

Class overrides: some products have a wrong product type in Amazon's metadata (for example a sofa
labelled CHAIR). The class overrides in `configs/data.yaml` (`data.abo.class_overrides`) map such a
product id to its correct class, and `select` uses that class instead of the class rules. Why a list
in the settings, and not an edit of the metadata files: the metadata is downloaded and not kept in
git, so an edit would be lost with a new download, and nobody could see what was changed or why.
The comments next to the list give the reason for each product.

To apply a changed list to downloaded data, run the download step again with the stages
"meshes photos catalog". The selection then gives the corrected products their new mesh ids; the
meshes and photos stages fetch their files into the folders of the new class (all other files are
skipped); and the catalog stage writes the new catalog rows. Files in the folder of the old class
are no longer used and can be deleted. Do this before the ABO meshes are rendered, because the
class is also part of the render folder.

Identifiers used by the project:

- A mesh is identified as "abo/<class>/<model id>", for example "abo/chair/B07B4FZN9H".
  The model id is the Amazon product id (ASIN) of the model.
- A photo is identified by its Amazon image id, for example "81iZlv3bjpL". The same photo can
  appear in several listings, so the reader keeps each image id once.

Axis convention: ABO meshes have +Y up and +Z to the front, the same as Pix3D. The units are metres.
"""

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATASET = "abo"
BUCKET_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"

# Metadata files: local file name -> path in the bucket.
METADATA_FILES = {
    "3dmodels.csv.gz": "3dmodels/metadata/3dmodels.csv.gz",
    "images.csv.gz": "images/metadata/images.csv.gz",
    **{f"listings_{h}.json.gz": f"listings/metadata/listings_{h}.json.gz" for h in "0123456789abcdef"},
}


def model_url(glb_path: str) -> str:
    """Return the bucket URL of a model's `.glb` file, for example ".../3dmodels/original/L/B01N2PLWIL.glb"."""
    return f"{BUCKET_URL}/3dmodels/original/{glb_path}"


def image_url(image_path: str, size: str = "original") -> str:
    """Return the bucket URL of a photo. size "original" = full resolution; "small" = at most 256 pixels."""
    return f"{BUCKET_URL}/images/{size}/{image_path}"


@dataclass
class AboMesh:
    """One selected ABO model: what the download step needs, and what the catalog gets later."""

    mesh_id: str  # "abo/<class>/<model id>"
    category: str  # our class, for example "chair"
    model_id: str  # Amazon product id of the model, for example "B07B4FZN9H"
    glb_path: str  # path of the .glb file in the bucket, for example "L/B01N2PLWIL.glb"
    product_type: str  # ABO product type, for example "CHAIR"
    leaf: str  # last part of the category path, for example "Chairs" ("" if the listing has none)
    num_vertices: int  # vertex count from 3dmodels.csv (all meshes of the file)
    num_faces: int  # face count from 3dmodels.csv
    extent: tuple[float, float, float]  # size of the model in metres along x, y, z, from 3dmodels.csv
    image_ids: list[str] = field(default_factory=list)  # the photos of all listings of the model, each once


@dataclass
class AboImage:
    """One catalog photo of the selected models."""

    image_id: str  # Amazon image id, for example "81iZlv3bjpL"
    category: str  # class of the first selected model that shows this photo
    path: str  # path in the bucket, below images/original/ (for example "14/14fe8812.jpg")
    width: int
    height: int
    mesh_ids: list[str] = field(default_factory=list)  # every selected mesh whose listings show this photo


def read_models(path: Path) -> dict[str, dict]:
    """Read 3dmodels.csv.gz: {model id: row}. The counts and the extent are converted to numbers."""
    with gzip.open(path, "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["3dmodel_id"]: {
            "glb_path": row["path"],
            "num_vertices": int(row["vertices"]),
            "num_faces": int(row["faces"]),
            "extent": (float(row["extent_x"]), float(row["extent_y"]), float(row["extent_z"])),
        }
        for row in rows
    }


def read_images(path: Path) -> dict[str, dict]:
    """Read images.csv.gz: {image id: {"path", "width", "height"}}."""
    with gzip.open(path, "rt", newline="") as f:
        return {
            row["image_id"]: {"path": row["path"], "width": int(row["width"]), "height": int(row["height"])}
            for row in csv.DictReader(f)
        }


def _leaf(listing: dict) -> str:
    """Return the last part of the first category path of a listing ("" if there is none)."""
    nodes = listing.get("node") or []
    if not nodes:
        return ""
    return nodes[0].get("node_name", "").rstrip("/").split("/")[-1]


def read_model_listings(paths: list[Path]) -> dict[str, dict]:
    """Read the listing files and combine the listings of each 3D model.

    Returns:
        {model id: {"product_type", "leaf", "image_ids"}}. The product type and the leaf come from
        the first listing of the model (in file order) that has them; the image ids are the main
        photo and the other photos of all its listings, each once, in order of first appearance.
    """
    combined: dict[str, dict] = {}
    for path in sorted(paths):  # listings_0 ... listings_f: a fixed order makes the result repeatable
        with gzip.open(path, "rt") as f:
            for line in f:
                listing = json.loads(line)
                model_id = listing.get("3dmodel_id")
                if not model_id:
                    continue  # most listings have no 3D model
                entry = combined.setdefault(model_id, {"product_type": "", "leaf": "", "image_ids": []})
                if not entry["product_type"] and listing.get("product_type"):
                    entry["product_type"] = listing["product_type"][0]["value"]
                if not entry["leaf"]:
                    entry["leaf"] = _leaf(listing)
                for image_id in [listing.get("main_image_id")] + list(listing.get("other_image_id") or []):
                    if image_id and image_id not in entry["image_ids"]:
                        entry["image_ids"].append(image_id)
    return combined


def class_of(product_type: str, leaf: str, rules: list[dict]) -> str | None:
    """Apply the class rules: return the class of the first rule that matches, or None.

    A rule matches when its "product_type" equals the product type and, if the rule has a "leaf"
    pattern, the pattern matches the leaf (regular expression, searched anywhere in the leaf).
    """
    for rule in rules:
        if rule["product_type"] != product_type:
            continue
        if "leaf" in rule and not re.search(rule["leaf"], leaf):
            continue
        return rule["class"]
    return None


def select(
    models: dict[str, dict],
    listings: dict[str, dict],
    images: dict[str, dict],
    rules: list[dict],
    max_models_per_class: int | None = None,
    class_overrides: dict[str, str] | None = None,
) -> tuple[list[AboMesh], list[AboImage]]:
    """Select the models of our classes and their photos.

    Args:
        models: From `read_models`.
        listings: From `read_model_listings`.
        images: From `read_images`.
        rules: The class rules (`data.abo.class_rules`).
        max_models_per_class: None = every model; otherwise the first N model ids of each class in
            sorted order.
        class_overrides: Product id -> correct class, for products with a wrong product type in the
            metadata (`data.abo.class_overrides`; see the module description). An override wins over
            the class rules. The AboMesh keeps Amazon's product type, so the correction stays visible.

    Returns:
        meshes: One AboMesh per selected model, sorted by mesh id.
        photos: One AboImage per distinct photo of the selected models, sorted by image id. Photo
            ids that are missing from images.csv are left out.

    Raises:
        ValueError: If an override names a product id without a 3D model, or a class that no class
            rule gives (both are most likely typing errors in the settings).
    """
    class_overrides = class_overrides or {}
    unknown_ids = sorted(set(class_overrides) - set(models))
    known_classes = {rule["class"] for rule in rules}
    unknown_classes = sorted(set(class_overrides.values()) - known_classes)
    if unknown_ids or unknown_classes:
        raise ValueError(f"class overrides: unknown product ids {unknown_ids}, unknown classes {unknown_classes}")

    by_class: dict[str, list[AboMesh]] = defaultdict(list)
    for model_id in sorted(models):
        listing = listings.get(model_id)
        if listing is None:
            continue
        # The override (if there is one) replaces the class that the rules give for Amazon's label.
        category = class_overrides.get(model_id) or class_of(listing["product_type"], listing["leaf"], rules)
        if category is None:
            continue
        model = models[model_id]
        by_class[category].append(
            AboMesh(
                mesh_id=f"{DATASET}/{category}/{model_id}",
                category=category,
                model_id=model_id,
                glb_path=model["glb_path"],
                product_type=listing["product_type"],
                leaf=listing["leaf"],
                num_vertices=model["num_vertices"],
                num_faces=model["num_faces"],
                extent=model["extent"],
                image_ids=[i for i in listing["image_ids"] if i in images],
            )
        )
    meshes = sorted(
        (m for group in by_class.values() for m in group[: max_models_per_class or None]), key=lambda m: m.mesh_id
    )
    photos: dict[str, AboImage] = {}
    for mesh in meshes:
        for image_id in mesh.image_ids:
            if image_id not in photos:
                info = images[image_id]
                photos[image_id] = AboImage(image_id, mesh.category, info["path"], info["width"], info["height"])
            photos[image_id].mesh_ids.append(mesh.mesh_id)
    return meshes, [photos[i] for i in sorted(photos)]


def class_counts(meshes: list[AboMesh], photos: list[AboImage]) -> dict[str, dict[str, int]]:
    """Return {class: {"meshes": n, "photos": n}} for a report, sorted by the number of meshes."""
    mesh_counts = Counter(m.category for m in meshes)
    photo_counts = Counter(p.category for p in photos)
    return {c: {"meshes": n, "photos": photo_counts[c]} for c, n in mesh_counts.most_common()}


# Manifest: the selection written as two CSV files, so that later steps do not read the metadata again.
MESH_MANIFEST_COLUMNS = ["mesh_id", "category", "model_id", "glb_path", "product_type", "leaf",
                         "num_vertices", "num_faces", "extent", "image_ids"]
IMAGE_MANIFEST_COLUMNS = ["image_id", "category", "path", "width", "height", "mesh_ids"]


def write_manifest(meshes: list[AboMesh], photos: list[AboImage], folder: Path) -> tuple[Path, Path]:
    """Write `meshes.csv` and `images.csv` into `folder`. Lists are joined with spaces."""
    folder.mkdir(parents=True, exist_ok=True)
    mesh_path, image_path = folder / "meshes.csv", folder / "images.csv"
    with open(mesh_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MESH_MANIFEST_COLUMNS)
        writer.writeheader()
        for m in meshes:
            writer.writerow({**m.__dict__, "extent": " ".join(f"{v:.6f}" for v in m.extent), "image_ids": " ".join(m.image_ids)})
    with open(image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=IMAGE_MANIFEST_COLUMNS)
        writer.writeheader()
        for p in photos:
            writer.writerow({**p.__dict__, "mesh_ids": " ".join(p.mesh_ids)})
    return mesh_path, image_path


def read_manifest(folder: Path) -> tuple[list[AboMesh], list[AboImage]]:
    """Read the two manifest files that `write_manifest` wrote."""
    with open(folder / "meshes.csv", newline="") as f:
        meshes = [
            AboMesh(
                mesh_id=r["mesh_id"], category=r["category"], model_id=r["model_id"], glb_path=r["glb_path"],
                product_type=r["product_type"], leaf=r["leaf"], num_vertices=int(r["num_vertices"]),
                num_faces=int(r["num_faces"]), extent=tuple(float(v) for v in r["extent"].split()),
                image_ids=r["image_ids"].split(),
            )
            for r in csv.DictReader(f)
        ]
    with open(folder / "images.csv", newline="") as f:
        photos = [
            AboImage(r["image_id"], r["category"], r["path"], int(r["width"]), int(r["height"]), r["mesh_ids"].split())
            for r in csv.DictReader(f)
        ]
    return meshes, photos
