import numpy as np
import pyvista as pv
from pathlib import Path


def render_prediction(
        prediction,
        patient_id):

    output = Path("results") / patient_id

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    p = pv.Plotter(
        off_screen=True
    )

    # Render each diagnostic class independently.  A single volume colormap
    # obscures class identities in the final image, whereas separate surfaces
    # make edema and tumour core immediately distinguishable.
    class_styles = (
        (1, "#00c8ff", "Edema (cyan)", 0.45),
        (2, "#ff6b00", "Tumor core (orange)", 0.75),
        (3, "#e53935", "Enhancing tumor (red)", 0.80),
    )
    for class_id, colour, label, opacity in class_styles:
        class_mask = (prediction == class_id).astype(np.uint8)
        if not np.any(class_mask):
            continue
        grid = pv.ImageData(dimensions=np.array(class_mask.shape) + 1)
        grid.cell_data["mask"] = class_mask.flatten(order="F")
        surface = grid.cell_data_to_point_data().contour([0.5], scalars="mask")
        if surface.n_cells:
            p.add_mesh(surface, color=colour, opacity=opacity, label=label)

    if np.any(prediction > 0):
        p.add_legend(
            bcolor="white",
            face="circle",
            size=(0.20, 0.12),
            loc="upper left",
        )
    else:
        p.add_text(
            "No tumor detected",
            position="upper_left",
            font_size=18,
            color="black",
        )

    p.view_isometric()
    p.set_background("white")

    p.show(
        screenshot=str(
            output/"prediction3d.png"
        )
    )

    p.close()
