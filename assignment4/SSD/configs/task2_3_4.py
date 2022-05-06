from ssd.modeling import FocalLoss
from tops.config import LazyCall as L
from ssd.modeling import SSD300
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

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes="${train.num_classes}",
    use_better_weights = True
)
