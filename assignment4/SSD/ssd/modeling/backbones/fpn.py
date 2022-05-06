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
#         self.backbone.layer5 = torch.nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(in_channels=output_channels[3],out_channels=256,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256,out_channels=self.out_channels[4],kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#         )
#         self.backbone.layer6 = torch.nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(in_channels=self.out_channels[4],out_channels=128,kernel_size=1,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128,out_channels=self.out_channels[5],kernel_size=1,stride=2,padding=0),
#             nn.ReLU(),
#         )
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
        
        #layer1 = self.backbone.maxpool(x)
        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        layer5 = self.backbone.layer5(layer4)
        layer6 = self.backbone.layer6(layer5)
        
# #         print(self.backbone)

#         featureDict = dict([("x0",x),("x1",layer2),("x2",layer3),("x3",layer4),("x4",layer5),("x5",layer6)])

#         fpn_output = self.fpn(featureDict)
        
#         outprrrt = [x, layer2, layer3, layer4, layer5, layer6, fpn_output]
        
#         return tuple(fpn_output.values())
        fpn_input_features = {}
        fpn_input_features["0"] = layer1
        fpn_input_features["1"] = layer2
        fpn_input_features["2"] = layer3
        fpn_input_features["3"] = layer4
        fpn_input_features["4"] = layer5
        fpn_input_features["5"] = layer6
        
        fpn_output = self.fpn(fpn_input_features)
        
#         out_features = OrderedDict([
#             ("0", fpn_output["0"]), 
#             ("1", fpn_output["1"]), 
#             ("2", fpn_output["2"]), 
#             ("3", fpn_output["3"]), 
#             ("4", fpn_output["4"]),
#             ("pool", fpn_output["5"])
#         ])

        return fpn_output.values()