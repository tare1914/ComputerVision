from cv2 import _OutputArray_DEPTH_MASK_FLT
import torch
from torch import nn
from typing import Tuple, List

from torchvision.models.resnet import BasicBlock
from torchvision.ops import FeaturePyramidNetwork 

class FPN(torch.nn.Module):
    def __init__(self,
            input_channels: List[int],
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        self.backbone = torchvision.models.resnet18(pretrained=True)
#         self.backbone = torchvision.models.resnet34(pretrained=True)
#         self.backbone = torchvision.models.resnet50(pretrained=True)
#         self.backbone = torchvision.models.resnet101(pretrained=True)
#         self.backbone = torchvision.models.resnet152(pretrained=True)

        


        self.pyramid = FeaturePyramidNetwork(
            in_channels_list=self.out_channels, 
            out_channels=output_channels
        )
    def forward(self, x):
        out_features = self.pyramid.
        y = x
        for feature in self.pyr:
            y = feature(y)
            out_features.append(y)