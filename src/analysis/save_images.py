import matplotlib.pyplot as plt


def save_middle_slice(volume, filename):
    """
    Save the middle slice of a 3D volume.

    Parameters
    ----------
    volume : numpy.ndarray
        Shape (Depth, Height, Width)

    filename : Path or str
    """

    # Preprocessed model volumes use (Height, Width, Depth).
    middle = volume.shape[2] // 2

    plt.figure(figsize=(6, 6))

    plt.imshow(volume[:, :, middle], cmap="gray")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()
