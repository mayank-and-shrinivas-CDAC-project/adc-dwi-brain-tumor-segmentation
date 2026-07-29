"""Backward-compatible model loading helpers.

The actual model definitions and checkpoint registry live in ``src.models``.
"""

import torch

from src.models.architectures import create_swin_unetr, create_unet


def load_unet(weight_path, device):
    model = create_unet(device)
    state = _load_checkpoint(weight_path, device)
    model.load_state_dict(state)
    model.eval()

    return model


def load_swin(weight_path, device):
    model = create_swin_unetr(device)
    state = _load_checkpoint(weight_path, device)
    model.load_state_dict(state)
    model.eval()

    return model


def _load_checkpoint(weight_path, device):
    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    return checkpoint
