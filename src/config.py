"""
Centralized configuration for all thesis notebooks.

Usage:
    from src.config import get_paths, CONSTANTS
    paths = get_paths()  # auto-detects Colab vs local
"""
import os
from pathlib import Path


# ─── Constants ───────────────────────────────────────────────────────
IMG_SIZE = 256
NUM_CLASSES = 3
CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
MAX_PIXEL_VALUE = 255
RANDOM_SEED = 42

# Training defaults
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_KFOLDS = 5


def is_colab():
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_paths(drive_root=None):
    """
    Return a dict of all project paths, auto-detecting environment.

    On Colab:  drive_root defaults to /content/drive/MyDrive/TESIS
    Locally:   drive_root defaults to the repo root (parent of src/)
    """
    if drive_root is None:
        if is_colab():
            drive_root = Path("/content/drive/MyDrive/TESIS")
        else:
            # Assume running from repo root
            drive_root = Path(__file__).resolve().parent.parent

    drive_root = Path(drive_root)

    return {
        # ── Root ──
        "root": drive_root,

        # ── Data splits ──
        "splits": drive_root / "data" / "splits",
        "datasetclean_zip": drive_root / "data" / "splits" / "datasetclean.zip",
        "datasetcleanmask_zip": drive_root / "data" / "splits" / "datasetcleanmask.zip",
        "testreducido_zip": drive_root / "data" / "splits" / "testreducido.zip",
        "trainreducido_zip": drive_root / "data" / "splits" / "trainreducido.zip",

        # ── Synthetic images ──
        "synth_batches": drive_root / "data" / "synth_batches",

        # ── Features ──
        "features": drive_root / "data" / "features",
        "w_features": drive_root / "data" / "features" / "w_features",
        "projections": drive_root / "data" / "projections_batch_all",

        # ── Sample data ──
        "samples": drive_root / "data" / "samples",
        "archive2": drive_root / "data" / "archive2",
        "testingset": drive_root / "data" / "Testingset",
        "counterfactual_pairs": drive_root / "data" / "counterfactual_pairs",
        "prototypes": drive_root / "data" / "prototypes",

        # ── Models ──
        "models": drive_root / "models",
        "stylegan_pkl_900": drive_root / "models" / "stylegan2" / "network-snapshot-000900.pkl",
        "stylegan_pkl_600": drive_root / "models" / "stylegan2" / "network-snapshot-000600.pkl",
        "stylegan_rosinality": drive_root / "models" / "stylegan_brain_rosinality.pt",
        "encoder_e4e": drive_root / "models" / "encoder_e4e_best_model.pt",
        "cnn_resnet": drive_root / "models" / "cnn_resnet_best.pt",
        "encoder_sup": drive_root / "models" / "encoder_sup_best.pt",
        "cnn_checkpoints": drive_root / "models" / "cnn_checkpoints",

        # ── Results ──
        "results": drive_root / "results",
        "results_exp1": drive_root / "results" / "exp1_generation",
        "results_exp2a": drive_root / "results" / "exp2a_counterfactuals",
        "results_exp2b": drive_root / "results" / "exp2b_latent_inversion",
        "results_exp2c": drive_root / "results" / "exp2c_augmentation",
        "results_exp3": drive_root / "results" / "exp3_hitl",

        # ── Libraries ──
        "stylegan2_lib": drive_root / "stylegan2-ada-pytorch",
        "encoder4editing_lib": drive_root / "encoder4editing",
    }
