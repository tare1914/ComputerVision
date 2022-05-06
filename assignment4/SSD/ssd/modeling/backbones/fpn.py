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

        self.backbone.layer5 = torch.nn.Sequential(
            BasicBlock(inplanes=output_channels[-3], planes=output_channels[-2], stride = 2, downsample=nn.Sequential(
                nn.Conv2d(output_channels[-3], output_channels[-2], kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(output_channels[-2])
                )
            )
        )

        self.backbone.layer6 = torch.nn.Sequential(
            BasicBlock(inplanes=output_channels[-2], planes=output_channels[-1], stride=2, downsample=nn.Sequential(
                nn.Conv2d(output_channels[-2], output_channels[-1], kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(output_channels[-1])
                )
            )
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

        fpn_in = {}
        fpn_in["0"] = x
        fpn_in["1"] = layer2
        fpn_in["2"] = layer3
        fpn_in["3"] = layer4
        fpn_in["4"] = layer5
        fpn_in["5"] = layer6

        fpn_out = self.fpn(fpn_in)

        output = [x, layer2, layer3, layer4, layer5, layer6]

        return output