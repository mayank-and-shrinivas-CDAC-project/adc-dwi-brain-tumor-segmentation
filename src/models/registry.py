"""Checkpoint registry and safe model loading for the desktop demo."""

from pathlib import Path

import torch

from .architectures import create_swin_unetr, create_unet


MODEL_DIR = Path(__file__).resolve().parent

MODEL_SPECS = {
    "GE_SW_ENT": (create_swin_unetr, "swin_unetr_tuned_ge_best.pt"),
    "SIEMENS_SW_ENT": (create_swin_unetr, "swin_unetr_tuned_siemens_best.pt"),
    "GE_UNET": (create_unet, "unet_tuned_ge_best.pt"),
    "SIEMENS_UNET": (create_unet, "unet_tuned_siemens_best.pt"),
}


def available_models():
    """Return model names in the same scanner/model order used by the UI."""
    return tuple(MODEL_SPECS)


def checkpoint_path(model_name):
    """Return the packaged checkpoint path or raise a useful error."""
    try:
        _, filename = MODEL_SPECS[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model: {model_name}") from error
    path = MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model weights were not found: {path}")
    return path


def _state_dict(checkpoint):
    """Accept plain PyTorch state dicts and common training checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model(model_name, device="cpu"):
    """Create a packaged model and load its matching checkpoint strictly."""
    try:
        factory, _ = MODEL_SPECS[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model: {model_name}") from error

    path = checkpoint_path(model_name)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(path, map_location=device)

    model = factory(device)
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    model.eval()
    return model
