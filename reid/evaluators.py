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

def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for _, f, _, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for _, f, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def extract_cnn_feature(model, inputs, typess="query"):
    inputs_h = inputs
    MaskNet, TaskNet = model
    MaskNet.eval()
    TaskNet.eval()

    inputs_h = to_torch(inputs_h)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs_h = inputs_h.to(device)
    with torch.no_grad():
        f1_h = TaskNet(inputs_h, types="encoder", drop=False)
        mask_h, score = MaskNet(f1_h) #+ 0.5
        # 
        # import cv2
        # for i, (inputs_k, mask_k) in enumerate(zip(inputs_h, mask_h)):
        #     # if i == 10:
        #     #     break
        #     k = inputs_k.cpu().detach().numpy().transpose(1,2,0)[:,:,::-1]
        #     k = (k - k.min()) / (k.max()-k.min()) * 255
        #     mask_k = mask_k.cpu().detach().numpy()
        #     mask_k = (mask_k - mask_k.min()) / (mask_k.max()-mask_k.min()) * 255
        #     mask_k = np.array(np.mean(mask_k,axis=0), np.uint8)
        #     mask_k = cv2.applyColorMap(mask_k, cv2.COLORMAP_JET)
        #     mask_k = cv2.resize(mask_k, (128, 256))
        #     k = k*0.4 + mask_k * 0.6
        #     cv2.imwrite("imgs/occludedduke/mask_" + str(i) + ".jpg", k)
        # exit()
        # 
        score = torch.nn.functional.softmax(score, dim=1)
        # score_head, score_body, score_leg, score_shose = score_part
        outputs = TaskNet(f1_h * mask_h, types="tasknet", test=True)
        # import pdb; pdb.set_trace()

        # outputs = TaskNet(f1_h, types="tasknet", test=True)
        outputs = outputs.data.cpu()
    return outputs



def extract_features(model, data_loader, print_freq=1, types="query"):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        outputs = extract_cnn_feature(model, imgs, typess=types)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

    return features, labels
# 

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
        query_features, _ = extract_features(self.model, query_loader, 1, types='query')
        gallery_features, _ = extract_features(self.model, gallery_loader, 1, types='gallery')
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat = pairwise_distance(query_features, gallery_features, query, gallery)
            if save_dir is not None:
                visualize_ranked_results(distmat, query, gallery, save_dir)
        return evaluate_all(distmat, query=query, gallery=gallery)

