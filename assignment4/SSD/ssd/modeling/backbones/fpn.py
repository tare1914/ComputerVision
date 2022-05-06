from cv2 import _OutputArray_DEPTH_MASK_FLT
import torch
import torchvision
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
        
#         self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone = torchvision.models.resnet34(pretrained=True)
#         self.backbone = torchvision.models.resnet50(pretrained=True)
#         self.backbone = torchvision.models.resnet101(pretrained=True)
#         self.backbone = torchvision.models.resnet152(pretrained=True)

        print(self.out_channels)
        self.backbone.layer5 = torch.nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels[3],out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=self.out_channels[4],kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
        )
        self.backbone.layer6 = torch.nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[4],out_channels=128,kernel_size=1,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=self.out_channels[5],kernel_size=1,stride=2,padding=0),
            nn.ReLU(),
        )

        self.fpn = FeaturePyramidNetwork(
            in_channels_list = output_channels,
            out_channels = image_channels
        )

    def forward(self, x1):

        x = self.backbone.conv1(x1)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        layer5 = self.backbone.layer5(layer4)
        layer6 = self.backbone.layer6(layer5)
        
        print(self.backbone)

        #featureDict = dict([("x0",x12),("x1",layer2),("x2",layer3),("x3",layer4),("x4",layer5),("x5",layer6)])

        outprrrt = [x, layer2, layer3, layer4, layer5, layer6]

        return outprrrt