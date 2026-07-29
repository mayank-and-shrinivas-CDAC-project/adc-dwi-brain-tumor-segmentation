import numpy as np
from scipy.ndimage import label

TUMOR_PRESENT_MIN_VOXELS = 50


def analyze_prediction(prediction, brain_mask=None):

    """
    prediction:
        numpy array
        Shape (H,W,D)

    Returns
    -------
    dict
    """

    report = {}

    if brain_mask is None:
        brain_voxels = prediction.size
    else:
        brain_mask = np.asarray(brain_mask, dtype=bool)
        if brain_mask.shape != prediction.shape:
            raise ValueError("Brain-mask shape does not match the prediction.")
        brain_voxels = int(brain_mask.sum())

    tumor = prediction > 0

    tumor_voxels = int(tumor.sum())

    report["brain_voxels"] = int(brain_voxels)
    report["tumor_voxels"] = tumor_voxels
    report["tumor_present"] = tumor_voxels >= TUMOR_PRESENT_MIN_VOXELS

    report["tumor_percent"] = (
        tumor_voxels / brain_voxels * 100
    )

    # -----------------------------
    # Connected components
    # -----------------------------

    labeled, num = label(tumor)

    report["components"] = num

    if num > 0:

        sizes = []

        for i in range(1, num + 1):

            sizes.append(
                np.sum(labeled == i)
            )

        report["largest_component"] = int(max(sizes))

    else:

        report["largest_component"] = 0

    # -----------------------------
    # Slice containing most tumor
    # -----------------------------

    tumor_per_slice = tumor.sum(axis=(0, 1))

    report["max_slice"] = int(
        np.argmax(tumor_per_slice)
    )

    # -----------------------------
    # Class statistics
    # -----------------------------

    report["background"] = int(
        np.sum(prediction == 0)
    )

    report["edema"] = int(
        np.sum(prediction == 1)
    )

    report["tumor_core"] = int(
        np.sum(prediction == 2)
    )

    report["enhancing"] = int(
        np.sum(prediction == 3)
    )

    return report

def create_report(patient_id,
                  model_name,
                  report):

    lines = []

    lines.append("=" * 50)
    lines.append("Brain Tumor Analysis Report")
    lines.append("=" * 50)
    lines.append("")

    lines.append(f"Patient ID : {patient_id}")
    lines.append(f"Model      : {model_name}")
    lines.append(
        "Tumor Present : " + ("YES" if report["tumor_present"] else "NO / NOT DETECTED")
    )

    lines.append("")
    lines.append("Volumes")
    lines.append("----------------------")

    lines.append(
        f"Brain Voxels : {report['brain_voxels']:,}"
    )

    lines.append(
        f"Tumor Voxels : {report['tumor_voxels']:,}"
    )

    lines.append(
        f"Tumor %      : {report['tumor_percent']:.3f}"
    )

    lines.append("")
    lines.append("Tumor")

    lines.append("----------------------")

    lines.append(
        f"Components : {report['components']}"
    )

    lines.append(
        f"Largest    : {report['largest_component']}"
    )

    lines.append(
        f"Max Slice  : {report['max_slice']}"
    )

    lines.append("")
    lines.append("Class Statistics")

    lines.append("----------------------")

    lines.append(
        f"Background : {report['background']:,}"
    )

    lines.append(
        f"Edema      : {report['edema']:,}"
    )

    lines.append(
        f"Tumor Core : {report['tumor_core']:,}"
    )

    lines.append(
        f"Enhancing  : {report['enhancing']:,}"
    )

    return "\n".join(lines)

from pathlib import Path


def save_report(text,
                patient_id):

    out = Path("reports")

    out.mkdir(exist_ok=True)

    filename = out / f"{patient_id}_report.txt"

    with open(filename,
              "w") as f:

        f.write(text)

    return filename
