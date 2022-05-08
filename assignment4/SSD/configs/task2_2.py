from tops.config import LazyCall as L
import torchvision
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors, RandomSampleCrop,
    RandomHorizontalFlip, RandomBrightness, RandomContrast)
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

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomBrightness),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])