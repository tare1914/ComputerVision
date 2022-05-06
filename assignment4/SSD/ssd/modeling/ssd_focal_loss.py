import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from tops import to_cuda

def focal_loss(outputs, targets, gamma = 2):
    
    alpha = to_cuda(torch.tensor([10,1000,1000,1000,1000,1000,1000,1000,1000]))
    soft = torch.permute(F.softmax(outputs,dim=1),(0,2,1))
    log_soft = torch.permute(F.log_softmax(outputs,dim=1),(0,2,1))
    targets = F.one_hot(targets,9)
    
    loss = -torch.sum(alpha*(1-soft)**gamma*targets*log_soft)/targets.size(dim=1)

    return loss

class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
       
        classification_loss = focal_loss(confs, gt_labels)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        
        total_loss = regression_loss/num_pos + classification_loss
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=classification_loss,
            total_loss=total_loss
        )
        return total_loss, to_log
