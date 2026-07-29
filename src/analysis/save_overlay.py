import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from pathlib import Path


def save_overlay(adc,
                 prediction,
                 patient_id):

    output_dir = Path("results") / patient_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Review the slice with the most predicted lesion rather than an arbitrary
    # anatomical middle slice.  This matches the notebook's lesion-first review.
    lesion_per_slice = (prediction > 0).sum(axis=(0, 1))
    middle = int(lesion_per_slice.argmax()) if lesion_per_slice.any() else prediction.shape[2] // 2

    figure, axis = plt.subplots(figsize=(6, 6))

    axis.imshow(adc[:, :, middle], cmap="gray")

    # Notebook convention: edema is cyan and tumour core is orange.  Keep the
    # fourth model class visible as red when a checkpoint produces it.
    colors = ListedColormap(["#00000000", "#00c8ff", "#ff6b00", "#e53935"])
    overlay = prediction[:, :, middle]
    axis.imshow(overlay, alpha=(overlay > 0) * 0.7, cmap=colors, vmin=0, vmax=3)
    axis.set_title("Segmentation overlay", pad=8)
    figure.legend(
        handles=[
            Patch(color="#00c8ff", label="Edema"),
            Patch(color="#ff6b00", label="Tumor core"),
            Patch(color="#e53935", label="Enhancing"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    axis.axis("off")

    figure.subplots_adjust(bottom=0.10, top=0.92)

    plt.savefig(
        output_dir/"overlay.png",
        dpi=200
    )

    plt.close(figure)
