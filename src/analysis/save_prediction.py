import numpy as np
import nibabel as nib
from pathlib import Path


def save_prediction(prediction, patient_id):

    output_dir = Path("results") / patient_id

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "prediction.npy",
        prediction.astype(np.uint8)
    )

    affine = np.eye(4)

    nii = nib.Nifti1Image(
        prediction.astype(np.uint8),
        affine
    )

    nib.save(
        nii,
        output_dir / "prediction.nii.gz"
    )

    return output_dir