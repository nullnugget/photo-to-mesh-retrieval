"""Every folder and file location of the v2 pipeline, defined in one place.

All paths are absolute and come from the repository root. Other modules import
these constants and do not build project paths themselves.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Settings and third-party model code
CONFIGS_DIR = ROOT / "configs"
THIRD_PARTY_DIR = ROOT / "third_party"

# Raw data (downloads, never modified)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PIX3D_DIR = RAW_DIR / "pix3d"
PIX3D_JSON = PIX3D_DIR / "pix3d.json"
ABO_DIR = RAW_DIR / "abo"
ABO_METADATA_DIR = ABO_DIR / "metadata"
ABO_IMAGES_DIR = ABO_DIR / "images"
ABO_MESHES_DIR = ABO_DIR / "meshes"

# Derived data (everything that v2 generates from the raw data)
DERIVED_DIR = DATA_DIR / "derived"
CATALOG_DIR = DERIVED_DIR / "catalog"
QUERIES_CSV = CATALOG_DIR / "queries.csv"
SHAPES_CSV = CATALOG_DIR / "shapes.csv"
QUERIES_DIR = DERIVED_DIR / "queries"
MASKS_DIR = DERIVED_DIR / "masks"
RENDERS_DIR = DERIVED_DIR / "renders"
RENDER_CONFIG_FILE = RENDERS_DIR / "render_config.yaml"
LIGHTING_TEST_DIR = DERIVED_DIR / "renders_lighting_test"
FEATURES_DIR = DERIVED_DIR / "features"

# Split files (small, tracked in git)
SPLITS_DIR = DATA_DIR / "splits"

# Pretrained weights and our checkpoints
WEIGHTS_DIR = ROOT / "weights"
CHECKPOINTS_DIR = WEIGHTS_DIR / "checkpoints"

# Run records and stand-alone check outputs
RUNS_DIR = ROOT / "runs"
REGISTRY_CSV = RUNS_DIR / "registry.csv"
TEST_OUTPUTS_DIR = ROOT / "test_outputs"

FEATURE_KINDS = ("renders", "queries")


def features_file(backbone: str, kind: str) -> Path:
    """Return the feature store file for one backbone, e.g. `dinov3_vitb16_renders.npz`."""
    if kind not in FEATURE_KINDS:
        raise ValueError(f"kind must be one of {FEATURE_KINDS}, got {kind!r}")
    return FEATURES_DIR / f"{backbone}_{kind}.npz"


def resolve(path: str | Path) -> Path:
    """Return an absolute path. A relative path is taken relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def ensure_dir(path: Path) -> Path:
    """Make the folder (and its parents) if it does not exist, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
