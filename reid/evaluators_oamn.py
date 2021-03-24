import time
from collections import OrderedDict
import pdb

import torch
import numpy as np

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

from torch.autograd import Variable
from .utils import to_torch
from .utils import to_numpy
import pdb

import os, math
from torchvision.utils import make_grid, save_image

import os.path as osp
import shutil
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from PIL import Image
from .models import resnet
import torch.nn as nn

import cv2
import torch.nn.functional as F
from .models import upsample

def pairwise_distance_(query_features, gallery_features, query=None, gallery=None, occlude=None):
    x = query_features
    y_l, y_r, y_d, y_u, y_l_, y_r_, y_d_, y_u_, y_h = gallery_features
    m, n = x.size(0), y_h.size(0)
    x = x.view(m, -1)
    y_l = y_l.view(n, -1)
    y_r = y_r.view(n, -1)
    y_d = y_d.view(n, -1)
    y_u = y_u.view(n, -1)
    y_l_ = y_l_.view(n, -1)
    y_r_ = y_r_.view(n, -1)
    y_d_ = y_d_.view(n, -1)
    y_u_ = y_u_.view(n, -1)
    y_h = y_h.view(n, -1)

    dist_l = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_l, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_l.addmm_(1, -2, x, y_l.t())
    dist_r = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_r, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_r.addmm_(1, -2, x, y_r.t())
    dist_d = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_d, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_d.addmm_(1, -2, x, y_d.t())
    dist_u = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_u, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_u.addmm_(1, -2, x, y_u.t())
    dist_l_ = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_l_, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_l_.addmm_(1, -2, x, y_l_.t())
    dist_r_ = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_r_, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_r_.addmm_(1, -2, x, y_r_.t())
    dist_d_ = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_d_, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_d_.addmm_(1, -2, x, y_d_.t())
    dist_u_ = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_u_, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_u_.addmm_(1, -2, x, y_u_.t())
    dist_h = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y_h, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_h.addmm_(1, -2, x, y_h.t())

    dist = torch.cat([dist_l.unsqueeze(0), dist_r.unsqueeze(0), dist_d.unsqueeze(0), dist_u.unsqueeze(0),dist_l_.unsqueeze(0), dist_r_.unsqueeze(0), dist_d_.unsqueeze(0), dist_u_.unsqueeze(0), dist_h.unsqueeze(0)], 0)
    occlude = torch.cat(occlude, 0).cpu().data
    locclude = torch.zeros(occlude.shape[0], 9).scatter_(1, occlude[:, None], 1).t().unsqueeze(2)
    dist = torch.sum(dist * locclude, dim=0)
    return dist

def extract_cnn_feature(model, inputs, typess="query", gallery=None):
    MaskNet, TaskNet = model
    MaskNet.eval()
    TaskNet.eval()

    inputs = to_torch(inputs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])
    if typess == "query":
        with torch.no_grad():
            f1 = TaskNet(inputs, types="encoder")
            mask, score = MaskNet(f1) #+ 0.5
            f1 = f1.reshape(f1.shape[0]//9, 9, f1.shape[1], f1.shape[2], f1.shape[3])
            mask = mask.reshape(mask.shape[0]//9, 9, mask.shape[1], mask.shape[2], mask.shape[3])
            score = score.reshape(score.shape[0]//9, 9, score.shape[1])
            score = score[:,0,:]
            
            occlude = torch.nn.functional.softmax(score, dim=1)
            occludedscore, occludedtype = torch.max(occlude.data.cpu(), dim=1)
            
            outputs = TaskNet(f1[:,0,:,:,:] * mask[:,0,:,:,:], types="tasknet", test=True).data.cpu()

            for i in range(f1.shape[0]):
                if occludedscore[i] <= 0.5:
                    occludedtype[i] = 8
                elif occludedscore[i] <= 0.75:
                    occludedtype[i] += 4

        return outputs, occludedtype
    else:
        f1 = TaskNet(inputs, types="encoder")
        mask, _ =  MaskNet(f1) #+ 0.5
        f1 = f1.reshape(f1.shape[0]//9, 9, f1.shape[1], f1.shape[2], f1.shape[3])
        f1_h = f1[:,0,:,:,:]
        f1_l = f1[:,1,:,:,:]
        f1_r = f1[:,2,:,:,:]
        f1_d = f1[:,3,:,:,:]
        f1_u = f1[:,4,:,:,:]
        f1_l_ = f1[:,5,:,:,:]
        f1_r_ = f1[:,6,:,:,:]
        f1_d_ = f1[:,7,:,:,:]
        f1_u_ = f1[:,8,:,:,:]

        mask = mask.reshape(mask.shape[0]//9, 9, mask.shape[1], mask.shape[2], mask.shape[3])
        mask_h = mask[:,0,:,:,:]

        outputs_h = TaskNet(f1_h * mask[:,0,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_l = TaskNet(f1_h * mask[:,1,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_r = TaskNet(f1_h * mask[:,2,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_d = TaskNet(f1_h * mask[:,3,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_u = TaskNet(f1_h * mask[:,4,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_l_ = TaskNet(f1_h * mask[:,5,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_r_ = TaskNet(f1_h * mask[:,6,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_d_ = TaskNet(f1_h * mask[:,7,:,:,:], types="tasknet", test=True).data.cpu()
        outputs_u_ = TaskNet(f1_h * mask[:,8,:,:,:], types="tasknet", test=True).data.cpu()

        return [outputs_l, outputs_r, outputs_d, outputs_u, outputs_l_, outputs_r_, outputs_d_, outputs_u_, outputs_h]

def extract_features(model, data_loader, print_freq=1, types="query", gallery=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = []
    g_features = []
    labels = []

    end = time.time()
    if types == "query":
        occludeds = []
        for i, (imgs, fnames, pids, _, _, _) in enumerate(data_loader):
            outputs, occludedtype = extract_cnn_feature(model, imgs, typess=types, gallery=gallery)
            features.append(outputs)
            labels.append(pids)
            occludeds.append(occludedtype)
        return torch.cat(features, 0), labels, occludeds
    else:
        features_h = []
        features_l = []
        features_r = []
        features_d = []
        features_u = []
        features_l_ = []
        features_r_ = []
        features_d_ = []
        features_u_ = []

        labels = []
        for i, (imgs, fnames, pids, _, _, _) in enumerate(data_loader):
            outputs = extract_cnn_feature(model, imgs, typess=types)
            features_l.append(outputs[0])
            features_r.append(outputs[1])
            features_d.append(outputs[2])
            features_u.append(outputs[3])
            features_l_.append(outputs[4])
            features_r_.append(outputs[5])
            features_d_.append(outputs[6])
            features_u_.append(outputs[7])
            features_h.append(outputs[8])
            labels.append(pids)
        return [torch.cat(features_l, 0), torch.cat(features_r, 0), torch.cat(features_d, 0), torch.cat(features_u, 0),torch.cat(features_l_, 0), torch.cat(features_r_, 0), torch.cat(features_d_, 0), torch.cat(features_u_, 0),torch.cat(features_h, 0)], labels

def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, _, pid, _, _ in query]
        gallery_ids = [pid for _, _, pid, _, _ in gallery]
        query_cams = [cam for _, _, _, cam, _ in query]
        gallery_cams = [cam for _, _, _, cam, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    return cmc_scores['market1501'][0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    # retrieve
    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None, rerank=False, save_dir=None):
        query_features, _, occludeds = extract_features(self.model, query_loader, 1, types='query')
        gallery_features,  _ = extract_features(self.model, gallery_loader, 1, types='gallery')
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat = pairwise_distance_(query_features, gallery_features, query, gallery, occludeds)
            if save_dir is not None:
                visualize_ranked_results(distmat, query, gallery, save_dir)
        return evaluate_all(distmat, query=query, gallery=gallery)

