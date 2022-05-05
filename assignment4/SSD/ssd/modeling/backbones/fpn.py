from cv2 import _OutputArray_DEPTH_MASK_FLT
import torch
from torch import nn
from typing import OrderedDict, Tuple, List

from torchvision.models.resnet import BasicBlock
from torchvision.ops import FeaturePyramidNetwork 

class FPN(torch.nn.Module):
    def __init__(self,
            #input_channels: List[int],
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

        self.backbone.layer5 = torch.nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[3],out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=self.out_channels[4],kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
        )
        self.backbone.layer6 = torch.nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[4],out_channels=128,kernel_size=2,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=self.out_channels[5],kernel_size=2,stride=2,padding=0),
            nn.ReLU(),
        )

        self.fpn = FeaturePyramidNetwork(
            output_channels,256
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(x)
        layer4 = self.backbone.layer4(x)
        layer5 = self.backbone.layer5(x)
        layer6 = self.backbone.layer6(x)

        featureDict = OrderedDict([("x1",x),("x2",layer2),("x3",layer3),("x4",layer4),("x5",layer5),("x6",layer6)])

        out_fpn = self.fpn(featureDict)

        return tuple(out_fpn.values())