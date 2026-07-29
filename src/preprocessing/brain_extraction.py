"""HD-BET brain extraction with the notebook's conservative local fallback."""

from pathlib import Path
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def robust_normalize(volume):
    volume = np.asarray(volume, dtype=np.float32)
    values = volume[np.isfinite(volume) & (volume > 0)]
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return (volume > 0).astype(np.float32)
    return np.clip((volume - low) / max(high - low, 1e-8), 0, 1)


def remove_eyes_and_orbit(mask):
    """Apply the notebook's conservative inferior eye/orbit removal."""
    mask = np.asarray(mask, dtype=bool).copy()
    height, width, depth = mask.shape
    inferior_cutoff = int(depth * 0.45)

    for z_index in range(inferior_cutoff):
        slice_mask = mask[:, :, z_index]
        if not np.any(slice_mask):
            continue
        rows = np.where(slice_mask.any(axis=1))[0]
        if not len(rows):
            continue

        row_min, row_max = rows.min(), rows.max()
        slice_height = row_max - row_min
        eye_zone_max = row_min + int(slice_height * 0.25)
        eroded = ndi.binary_erosion(slice_mask, structure=ndi.generate_binary_structure(2, 1), iterations=3)
        labels, count = ndi.label(eroded)
        if count:
            clean_eroded = np.zeros_like(eroded)
            for label in range(1, count + 1):
                component = labels == label
                if np.mean(np.where(component)[0]) > row_min + slice_height * 0.20:
                    clean_eroded |= component
            slice_mask &= ndi.binary_dilation(
                clean_eroded, structure=ndi.generate_binary_structure(2, 1), iterations=3
            )
        slice_mask[:eye_zone_max, :] = False
        mask[:, :, z_index] = slice_mask
    return mask.astype(np.uint8)


def _remove_eyes_in_dicom_order(mask):
    """Bridge DICOM (D,H,W) arrays to the notebook's (H,W,D) convention."""
    notebook_order = np.transpose(mask, (1, 2, 0))
    return np.transpose(remove_eyes_and_orbit(notebook_order), (2, 0, 1))


def fallback_brain_mask(volume):
    """Notebook-style conservative brain mask used when HD-BET is unavailable."""
    normalized = robust_normalize(volume)
    positive = normalized[normalized > 0]
    if positive.size < 10:
        return np.zeros(volume.shape, dtype=np.uint8)

    # Otsu is avoided here to keep the desktop demo usable without scikit-image.
    # The histogram implementation follows the same intent as the notebook:
    # segment brain-like tissue, retain only its principal 3D component, then
    # close small gaps before normalized ADC/DWI inference.
    histogram, edges = np.histogram(positive, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    weight_background = np.cumsum(histogram)
    weight_foreground = histogram.sum() - weight_background
    mean_background = np.cumsum(histogram * centers)
    mean_foreground = mean_background[-1] - mean_background
    valid = (weight_background > 0) & (weight_foreground > 0)
    between = np.zeros_like(centers)
    between[valid] = (
        (mean_background[valid] / weight_background[valid]
         - mean_foreground[valid] / weight_foreground[valid]) ** 2
        * weight_background[valid]
        * weight_foreground[valid]
    )
    threshold = max(float(centers[np.argmax(between)]) * 0.6, 0.05)
    mask = normalized > threshold
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask)
    if count:
        largest = 1 + np.argmax(np.bincount(labels.ravel())[1:])
        mask = labels == largest
    mask = ndi.binary_closing(mask, iterations=2)
    return _remove_eyes_in_dicom_order(mask)


def _run_hdbet(input_path, output_path):
    """Run whichever HD-BET interface is installed, returning its mask path."""
    try:
        from HD_BET.hd_bet_prediction import hdbet_predict

        hdbet_predict(
            input_files=[str(input_path)],
            output_files=[str(output_path)],
            device="cuda:0",
            save_mask=True,
        )
    except ImportError:
        try:
            from HD_BET.run import run_hd_bet

            run_hd_bet(
                input_path=str(input_path), output_path=str(output_path),
                mode="fast", device=0, postprocessing=True, save_mask=True,
            )
        except ImportError:
            executable = shutil.which("hd-bet")
            if executable is None:
                raise RuntimeError("HD-BET is not installed")
            subprocess.run(
                [executable, "-i", str(input_path), "-o", str(output_path), "-device", "0"],
                check=True,
            )

    candidates = (
        output_path.with_name(output_path.stem + "_mask.nii.gz"),
        output_path.with_name(output_path.stem + "_mask.nii"),
        output_path.parent / "extracted_brain_mask.nii.gz",
        output_path.parent / "extracted_brain_mask.nii",
    )
    return next((path for path in candidates if path.exists()), None)


def extract_brain_mask(volume):
    """Return ``(mask, method)``; HD-BET failures always safely use the fallback."""
    volume = np.asarray(volume, dtype=np.float32)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "input.nii.gz"
        output_path = temp_dir / "extracted_brain.nii.gz"
        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(input_path))
        try:
            mask_path = _run_hdbet(input_path, output_path)
            if mask_path is not None:
                mask = nib.load(str(mask_path)).get_fdata() > 0
                if mask.shape == volume.shape and np.any(mask):
                    return _remove_eyes_in_dicom_order(mask), "HD-BET"
        except Exception:
            pass
    return fallback_brain_mask(volume), "fallback brain-mask algorithm"
