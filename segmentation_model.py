import torch
import torch.nn as nn
from monai.networks.nets import UNet

def get_segmentation_model():
    """Return a U-Net model for brain segmentation."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,  # Example: gray matter, white matter, ventricles, hippocampus
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model
