from tops.config import LazyCall as L
from ssd.modeling.backbones import fpn
from ssd.modeling import AnchorBoxes

from .task2_3_2 import (
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

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[24, 24], [40, 40], [48, 48], [128, 128], [172, 172], [256, 256], [256, 800]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[1, 0.25], [1, 0.8, 0.25], [0.8, 0.25], [0.7, 0.25, 2], [3, 2, 4], [2, 3, 1.3, 4]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)


backbone = L(fpn.FPN)(
    output_channels=[64, 128, 256, 512, 256, 256],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

