from __future__ import absolute_import

import torch
from torch import cat
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb

from . import resnet
# from . import resnet_withinfo as resnet

from .rga_modules import RGA_Module




__all__ = ['resnet50']

class TaskNet(nn.Module):
    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(TaskNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.base = resnet.resnet50(pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.num_classes = num_classes
            out_planes = self.base.fc.in_features


            # Append new layers
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)
            self.drop = nn.Dropout(self.dropout)
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params() 
        

    def forward(self, x, types="tasknet", drop=False, test=False):
        
        if types == "encoder":
            x = self.base(x, types="encoder")
            return x
            
        x = self.base(x, types="tasknet")
        
        allfeature = x.clone()
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        triplet_feature = x
        x = self.feat_bn(x)
        if test:
            x = F.normalize(x)
            return x
        x = F.relu(x)
        x = self.drop(x)
        x_class = self.classifier(x)
        return x_class, triplet_feature, allfeature

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
class RGA_LAYER(nn.Module):
    def __init__(self, channel):
        super(RGA_LAYER, self).__init__()
        self.fc = RGA_Module(channel, (256//16)*(128//16), use_spatial=True, use_channel=False, cha_ratio=8, spa_ratio=8, down_ratio=8)
        self.score_bn = nn.BatchNorm1d(channel)
        init.constant_(self.score_bn.weight, 1)
        init.constant_(self.score_bn.bias, 0)
        # 
        # self.score_fc = nn.Linear(channel, 5)
        self.score_fc = nn.Linear(channel, 4)
        init.normal_(self.score_fc.weight, std=0.001)
        init.constant_(self.score_fc.bias, 0)

    def forward(self, x):
        b,c = x.shape[0], x.shape[1]
        mask = self.fc(x)
        score = self.score_fc(self.score_bn(F.avg_pool2d(x * mask.detach(), mask.size()[2:]).view(b,-1)))
        return mask, score
def resnet50(**kwargs):
    return RGA_LAYER(1024), TaskNet(50, **kwargs)