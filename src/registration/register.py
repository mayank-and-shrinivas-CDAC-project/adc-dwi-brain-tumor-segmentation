"""Notebook-equivalent ADC/DWI registration-ready alignment.

The reference notebook does not estimate a rigid transform.  It resizes the
DWI volume to the ADC grid with linear interpolation before preprocessing.
"""

import numpy as np
from scipy import ndimage as ndi


def register_volumes(adc_volume, dwi_volume):
    """Align DWI to ADC dimensions exactly as the reference notebook does."""

    adc = np.asarray(adc_volume, dtype=np.float32)
    dwi = np.asarray(dwi_volume, dtype=np.float32)

    if adc.ndim != 3 or dwi.ndim != 3:
        raise ValueError("ADC and DWI volumes must both be 3D.")

    if dwi.shape != adc.shape:
        zoom = [target / source for target, source in zip(adc.shape, dwi.shape)]
        dwi = ndi.zoom(dwi, zoom, order=1).astype(np.float32)

    return adc, dwi
