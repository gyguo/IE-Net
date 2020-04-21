# ------------------------------------------------------------------------------
# Author: Tao Zhao
# Descriptions:
# cfg.TRAIN.C_LOSS_NORM: coefficient for loss_norm, 0.0001 for THUMOS14
# note: fore_weights is [batch, channel, temporal_length] , different from original BaSNet, so dim=2 in
#       loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=2))
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


class BasNetLoss(nn.Module):
    def __init__(self):
        super(BasNetLoss, self).__init__()
        # self.l1loss = nn.L1Loss(reduction='mean')
        self.mseloss = nn.MSELoss(reduction='mean')
        # self.smoothl1loss = nn.SmoothL1Loss(reduction='mean')

    def _cls_loss(self, scores, labels):
        '''
        calculate classification loss
        1. dispose label, ensure the sum is 1
        2. calculate topk mean, indicates classification score
        3. calculate loss
        '''
        labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-10)
        clsloss = -torch.mean(torch.sum(labels * F.log_softmax(scores, dim=1), dim=1), dim=0)
        return clsloss

    def forward(self, score_cas, score_cam, labels, fore_weight):
        # labels_bg1 = torch.cat((labels, torch.ones((labels.shape[0], 1)).cuda()), dim=1)
        # labels= torch.cat((labels, torch.zeros((labels.shape[0], 1)).cuda()), dim=1)
        loss_cas = self._cls_loss(score_cas, labels)
        loss_cam = self._cls_loss(score_cam, labels)
        loss_norm = torch.mean(torch.norm(fore_weight, p=1, dim=1))
        # loss_consistency = self.l1loss(score_cas, score_cam)
        loss_consistency = self.mseloss(score_cas, score_cam)
        # loss_consistency = self.smoothl1loss(score_cas, score_cam)
        return loss_cas, loss_cam, loss_consistency, loss_norm
