"""Network factories for the ADC + DWI brain-tumour demo.

These definitions are kept next to the checkpoints so a model architecture and
its weights cannot silently drift apart.
"""

import inspect

from monai.networks.nets import SwinUNETR, UNet
from monai.networks.layers import Norm


NUM_CLASSES = 4
INPUT_CHANNELS = 2
INPUT_SHAPE = (128, 128, 96)


def create_unet(device="cpu"):
    """Build the architecture used by the tuned GE and Siemens U-Net weights."""
    return UNet(
        spatial_dims=3,
        in_channels=INPUT_CHANNELS,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        norm=Norm.INSTANCE,
        dropout=0.05,
    ).to(device)


def create_swin_unetr(device="cpu"):
    """Build the architecture used by the tuned GE and Siemens Swin UNETR weights."""
    kwargs = {
        "in_channels": INPUT_CHANNELS,
        "out_channels": NUM_CLASSES,
        "feature_size": 24,
        "use_checkpoint": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.10,
    }
    # MONAI versions before 1.3 require this constructor argument.
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = INPUT_SHAPE
    return SwinUNETR(**kwargs).to(device)
