"""
============================================================
Phase 4 - Preprocessing
------------------------------------------------------------
Prepares registered ADC/DWI volumes for MONAI inference.

Output tensor shape:
    (2,96,96,64)
============================================================
"""

import numpy as np
import torch
from scipy import ndimage as ndi
from src.preprocessing.brain_extraction import fallback_brain_mask

TARGET_SHAPE = (128, 128, 96)


def robust_normalize(volume, mask=None):
    """Notebook-equivalent 1st–99th percentile normalization."""
    volume = np.asarray(volume, dtype=np.float32)
    valid = np.isfinite(volume) & (volume > 0)
    if mask is not None and np.any(mask):
        values = volume[mask > 0]
    else:
        values = volume[valid]
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return ((volume > 0) if mask is None else (mask > 0)).astype(np.float32)
    normalized = np.clip((volume - lo) / max(hi - lo, 1e-8), 0, 1)
    if mask is not None:
        normalized *= mask > 0
    return normalized.astype(np.float32)


def resize_volume(volume, target_shape, order):
    zoom = [target / current for target, current in zip(target_shape, volume.shape)]
    return ndi.zoom(volume, zoom, order=order).astype(np.float32)


def preprocess(adc_volume, dwi_volume, brain_mask=None):

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # -------------------------------------------------------
    # Validate
    # -------------------------------------------------------

    if adc_volume.shape != dwi_volume.shape:
        raise ValueError("ADC and DWI shapes do not match.")

    print("✓ Validation Passed")

    # Convert raw DICOM order (Depth, Height, Width) to model order (H, W, D).
    adc = np.transpose(adc_volume, (1, 2, 0))
    dwi = np.transpose(dwi_volume, (1, 2, 0))
    if brain_mask is None:
        mask = fallback_brain_mask(adc)
    else:
        if brain_mask.shape != adc_volume.shape:
            raise ValueError("Brain-mask shape does not match the ADC volume.")
        mask = np.transpose(brain_mask, (1, 2, 0)).astype(np.uint8)
    adc = robust_normalize(adc, mask)
    dwi = robust_normalize(dwi, mask)
    mask = resize_volume(mask, TARGET_SHAPE, order=0) > 0.5
    adc = resize_volume(adc, TARGET_SHAPE, order=1) * mask
    dwi = resize_volume(dwi, TARGET_SHAPE, order=1) * mask
    image = torch.from_numpy(np.stack([adc, dwi], axis=0).astype(np.float32))

    print("✓ Brain-masked percentile normalization and resampling completed")

    print("After MONAI:", tuple(image.shape))

    # -------------------------------------------------------
    # Split channels for display
    # -------------------------------------------------------

    adc_processed = image[0].cpu().numpy()

    dwi_processed = image[1].cpu().numpy()

    print("ADC :", adc_processed.shape)
    print("DWI :", dwi_processed.shape)

    print("Tensor :", tuple(image.shape))

    print("=" * 60)

    return adc_processed, dwi_processed, image, mask.astype(np.uint8)
