from __future__ import print_function, absolute_import
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import pdb
import random
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import time


class BaseTrainer(object):
    def __init__(self, model, criterion, InvNet=None):
        super(BaseTrainer, self).__init__()
        self.MaskNet, self.TaskNet = model
        self.criterion = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.write = SummaryWriter(log_dir="./")
        self.index = []

    def train(self, epoch, data_loader, optimizer, batch_size, print_freq=1):
        self.MaskNet.train()
        self.TaskNet.train()

        optimizer_Mask, optimizer_Ide = optimizer

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_IDE_s = AverageMeter()
        losses_Tri = AverageMeter()
        losses_Mask1 = AverageMeter()
        losses_Mask2 = AverageMeter()
        losses_Mask3 = AverageMeter()
        losses_Score = AverageMeter()
        losses_Score2 = AverageMeter()

        train_loader = data_loader[0]
        end = time.time()

        for i, src_inputs in enumerate(train_loader):

            inputs, pids, pids_tri, masksize = self._parse_data(src_inputs)

            data_time.update(time.time() - end)
            end = time.time()

            loss_ce, loss_tri, loss_mask1, loss_mask3, loss_score = self._forward(inputs, pids, pids_tri, masksize, epoch)
           
            loss = loss_ce + 0.5 * loss_tri
            optimizer_Ide.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_Ide.step()

            loss_mask = loss + 5 * (loss_mask1 + loss_mask3) + 1 * loss_score
            optimizer_Mask.zero_grad()
            loss_mask.backward()
            optimizer_Mask.step()

            losses_IDE_s.update(loss_ce.item(), pids.size(0) * 4)
            losses_Tri.update(loss_tri.item(), pids.size(0) * 4)
            losses_Mask1.update(loss_mask1.item(), pids.size(0) * 4)
            losses_Mask3.update(loss_mask3.item(), pids.size(0) * 4)
            losses_Score.update(loss_score.item(), pids.size(0) * 4)

            batch_time.update(time.time() - end)
            end = time.time()


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}] \t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'task_1 {:.3f} ({:.3f})\t'
                      'tri {:.3f} ({:.3f})\t'
                      'mask1 {:.3f} ({:.3f})\t'
                      'mask2 {:.3f} ({:.3f})\t'
                      'mask3 {:.3f} ({:.3f})\t'
                      'score {:.3f} ({:.3f})\t'
                      'score2 {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(train_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_IDE_s.val, losses_IDE_s.avg,
                              losses_Tri.val, losses_Tri.avg,
                              losses_Mask1.val, losses_Mask1.avg,
                              losses_Mask2.val, losses_Mask2.avg,
                              losses_Mask3.val, losses_Mask3.avg,
                              losses_Score.val, losses_Score.avg,
                              losses_Score2.val, losses_Score2.avg))
    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs, tri=False):
        if tri:
            imgs, _, pids, _, pindexs = inputs
            inputs = imgs.to(self.device)
            pids = pids.to(self.device)
            return inputs, pids
        imgs, _, pids, _, pindexs, masksize = inputs
        inputs = imgs.to(self.device)
        inputs = inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4])
        pids = pids.to(self.device)
        pids_tri = pids.clone()
        pids = pids.unsqueeze(1).repeat(1, 5).view(-1)

        masksize = masksize.to(self.device)
        masksize[(128*256)//masksize == 2] = 1
        masksize[masksize!=1] = 0
        return inputs, pids, pids_tri, masksize

    def _forward(self, inputs, pids, pids_tri, masksize, epoch, update_only=False):
        f1 = self.TaskNet(inputs, types="encoder", drop=False)
        mask, score = self.MaskNet(f1.detach())
        outputs_source, triplet_feature, all_feature = self.TaskNet(f1 * mask, types="tasknet")
        loss_ce = self.criterion[0](outputs_source, pids)
        # 
        triplet_feature = triplet_feature.reshape(triplet_feature.shape[0]//5, 5, triplet_feature.shape[1])
        triplet_feature_h = triplet_feature[:,0,:]
        triplet_feature_l = triplet_feature[:,1,:]
        triplet_feature_r = triplet_feature[:,2,:]
        triplet_feature_d = triplet_feature[:,3,:]
        triplet_feature_u = triplet_feature[:,4,:]
        loss_tri = (self.criterion[1](triplet_feature_h, pids_tri) + self.criterion[1](triplet_feature_l, pids_tri) + self.criterion[1](triplet_feature_r, pids_tri) + self.criterion[1](triplet_feature_d, pids_tri) + self.criterion[1](triplet_feature_u, pids_tri)) / 5

        f1 = f1.reshape(f1.shape[0]//5, 5, f1.shape[1], f1.shape[2], f1.shape[3])
        f1_h = f1[:,0,:,:,:]
        f1_l = f1[:,1,:,:,:]
        f1_r = f1[:,2,:,:,:]
        f1_d = f1[:,3,:,:,:]
        f1_u = f1[:,4,:,:,:]

        mask = mask.reshape(mask.shape[0]//5, 5, mask.shape[1], mask.shape[2], mask.shape[3])
        mask_h = mask[:,0,:,:,:]
        mask_l = mask[:,1,:,:,:]
        mask_r = mask[:,2,:,:,:]
        mask_d = mask[:,3,:,:,:]
        mask_u = mask[:,4,:,:,:]

        f1_h_h, f1_h_d, f1_h_u, f1_h_l, f1_h_r = f1_h * mask_h, f1_h * mask_d, f1_h * mask_u, f1_h * mask_l, f1_h * mask_r
        f1_d_d, f1_u_u, f1_l_l, f1_r_r = f1_d * mask_d, f1_u * mask_u, f1_l * mask_l, f1_r * mask_r

        loss_mask1 = self.criterion[2](f1_h_l, f1_l_l) + self.criterion[2](f1_h_r, f1_r_r) + self.criterion[2](f1_h_d, f1_d_d) + self.criterion[2](f1_h_u, f1_u_u)
        masksize = masksize.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        loss_mask3 = self.criterion[2]((f1_l_l + f1_r_r) * masksize, f1_h_h * masksize) +  self.criterion[2]((f1_u_u + f1_d_d) * masksize, f1_h_h * masksize)

        # score_type1:
        score = score.reshape(score.shape[0]//5, 5, score.shape[1])
        score_h = score[:, 0, :]
        score_l = score[:, 1, :]
        score_r = score[:, 2, :]
        score_d = score[:, 3, :]
        score_u = score[:, 4, :]

        score_h = torch.max(torch.nn.functional.softmax(score_h.clone(), dim=1), dim=1)[0]
        score_l = torch.nn.functional.softmax(score_l.clone(), dim=1)[:,0]
        score_r = torch.nn.functional.softmax(score_r.clone(), dim=1)[:,1]
        score_d = torch.nn.functional.softmax(score_d.clone(), dim=1)[:,2]
        score_u = torch.nn.functional.softmax(score_u.clone(), dim=1)[:,3]

        masksize = masksize.squeeze(-1).squeeze(-1).squeeze(-1)
        id_h = torch.zeros(score_h.shape).cuda() + 0.25
        id_l = (torch.zeros(score_l.shape).cuda() + 1) * masksize + (torch.zeros(score_l.shape).cuda())* (1-masksize)
        id_r = (torch.zeros(score_r.shape).cuda() + 1) * masksize + (torch.zeros(score_r.shape).cuda())* (1-masksize)
        id_d = (torch.zeros(score_d.shape).cuda() + 1) * masksize + (torch.zeros(score_d.shape).cuda() + 0.5)* (1-masksize)
        id_u = (torch.zeros(score_u.shape).cuda() + 1) * masksize + (torch.zeros(score_u.shape).cuda() + 0.5)* (1-masksize)

        loss_score = self.criterion[2](score_h, id_h) + self.criterion[2](score_l, id_l) + self.criterion[2](score_r, id_r) + self.criterion[2](score_d, id_d) + self.criterion[2](score_u, id_u)
        return loss_ce, loss_tri, loss_mask1, loss_mask3, loss_score
