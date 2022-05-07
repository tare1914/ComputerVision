from ssd.modeling import FocalLoss
from tops.config import LazyCall as L
from ssd.modeling import SSD300
from .task2_3_1 import (
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

loss_objective = L(FocalLoss)(anchors="${anchors}")