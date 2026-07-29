"""
============================================================
Volume Builder
------------------------------------------------------------
Builds a 3D NumPy volume from a DICOM series.
============================================================
"""

import numpy as np
import pydicom


def diffusion_b_value(dataset):
    """Return a diffusion b-value when it is present in a DICOM dataset."""

    value = getattr(dataset, "DiffusionBValue", None)
    if value is None:
        # Siemens stores this value in a private tag in some exports.
        element = dataset.get((0x0019, 0x100C), None)
        value = getattr(element, "value", None)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_volume(series):
    """
    Build a 3D volume from a DICOM series.

    Parameters
    ----------
    series : list
        List returned by dicom_loader.py
        Each element:
            {
                "path": Path,
                "dataset": Dataset
            }

    Returns
    -------
    volume : np.ndarray
        Shape -> (Slices, Height, Width)
    """

    # A diffusion series may contain multiple acquisitions (commonly b=0 and
    # b=1000) at every slice location.  Stacking them together duplicates the
    # depth and causes alternating black-line artifacts after registration.
    # Use the highest available b-value, which is the diagnostic DWI series.
    b_values = [diffusion_b_value(item["dataset"]) for item in series]
    available_b_values = [value for value in b_values if value is not None]
    if available_b_values:
        selected_b_value = max(available_b_values)
        series = [
            item
            for item, b_value in zip(series, b_values)
            if b_value == selected_b_value
        ]

    # --------------------------------------------------------
    # Sort slices
    # --------------------------------------------------------

    def slice_position(item):
        position = getattr(item["dataset"], "ImagePositionPatient", None)
        if position is not None:
            try:
                return (0, float(position[2]))
            except (TypeError, ValueError, IndexError):
                pass
        return (1, int(getattr(item["dataset"], "InstanceNumber", 0)))

    series = sorted(series, key=slice_position)

    slices = []

    for item in series:

        ds = pydicom.dcmread(item["path"])

        image = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        slices.append(image)

    volume = np.stack(slices, axis=0)

    return volume
