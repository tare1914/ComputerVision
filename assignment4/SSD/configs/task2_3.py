from tops.config import LazyCall as L
import torchvision
import torchvision.models as models
from ssd.modeling.backbones import fpn

# The line belows inherits the configuration set for the tdt4265 dataset
from .base import (
    train,
    backbone,
    anchors,
    loss_objective,
    model,
    optimizer,
    schedulers,
    train_cpu_transform,
    val_cpu_transform,
    data_train,
    data_val,
    label_map
)

backbone = L(fpn.FPN)(
    output_channels=[64, 128, 256, 512, 256, 256],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

# model = L()(
#     feature_extractor="${backbone}",
#     anchors="${anchors}",
#     loss_objective="${loss_objective}",
#     num_classes=8 + 1  # Add 1 for background
# )