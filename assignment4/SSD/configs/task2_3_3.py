from tops.config import LazyCall as L
import torchvision
import torchvision.models as models
from ssd.modeling.backbones import fpn
from ssd.modeling import SSD_ChangedHead

# The line belows inherits the configuration set for the tdt4265 dataset
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


model = L(SSD_ChangedHead)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes="${train.num_classes}"
)