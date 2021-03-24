from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().view(1))
            dist_an.append(dist[i][mask[i] == 0].min().view(1))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap).data.float().mean()
        return loss

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

# import torch
# from torch import nn
# from torch.autograd import Variable
# import torch.nn.functional as F

# import numpy as np


# def normalize(x, axis=-1):
# 	"""Normalizing to unit length along the specified dimension.
# 	Args:
# 	  x: pytorch Variable
# 	Returns:
# 	  x: pytorch Variable, same shape as input
# 	"""
# 	x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
# 	return x

# def euclidean_dist(x, y):
# 	"""
# 	Args:
# 	  x: pytorch Variable, with shape [m, d]
# 	  y: pytorch Variable, with shape [n, d]
# 	Returns:
# 	  dist: pytorch Variable, with shape [m, n]
# 	"""
# 	m, n = x.size(0), y.size(0)
# 	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
# 	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
# 	dist = xx + yy
# 	dist.addmm_(1, -2, x, y.t())
# 	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
# 	return dist

# def cosine_dist(x, y):
# 	"""
# 	Args:
# 	  x: pytorch Variable, with shape [m, d]
# 	  y: pytorch Variable, with shape [n, d]
# 	"""
# 	x_normed = F.normalize(x, p=2, dim=1)
# 	y_normed = F.normalize(y, p=2, dim=1)
# 	return 1 - torch.mm(x_normed, y_normed.t())

# def cosine_similarity(x, y):
# 	"""
# 	Args:
# 	  x: pytorch Variable, with shape [m, d]
# 	  y: pytorch Variable, with shape [n, d]
# 	"""
# 	x_normed = F.normalize(x, p=2, dim=1)
# 	y_normed = F.normalize(y, p=2, dim=1)
# 	return torch.mm(x_normed, y_normed.t())


# def hard_example_mining(dist_mat, labels, return_inds=False):
# 	"""For each anchor, find the hardest positive and negative sample.
# 	Args:
# 	  dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
# 	  labels: pytorch LongTensor, with shape [N]
# 	  return_inds: whether to return the indices. Save time if `False`(?)
# 	Returns:
# 	  dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
# 	  dist_an: pytorch Variable, distance(anchor, negative); shape [N]
# 	  p_inds: pytorch LongTensor, with shape [N];
# 		indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
# 	  n_inds: pytorch LongTensor, with shape [N];
# 		indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
# 	NOTE: Only consider the case in which all labels have same num of samples,
# 	  thus we can cope with all anchors in parallel.
# 	"""
# 	assert len(dist_mat.size()) == 2
# 	assert dist_mat.size(0) == dist_mat.size(1)
# 	N = dist_mat.size(0)

# 	is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
# 	is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

# 	dist_ap, relative_p_inds = torch.max(
# 		dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
# 	dist_an, relative_n_inds = torch.min(
# 		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
	
# 	dist_ap = dist_ap.squeeze(1)
# 	dist_an = dist_an.squeeze(1)

# 	if return_inds:
# 		ind = (labels.new().resize_as_(labels)
# 			   .copy_(torch.arange(0, N).long())
# 			   .unsqueeze(0).expand(N, N))
# 		p_inds = torch.gather(
# 			ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
# 		n_inds = torch.gather(
# 			ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
# 		p_inds = p_inds.squeeze(1)
# 		n_inds = n_inds.squeeze(1)
# 		return dist_ap, dist_an, p_inds, n_inds

# 	return dist_ap, dist_an


# # ==============
# #  Triplet Loss 
# # ==============
# class TripletHardLoss(object):
# 	"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
# 	Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
# 	Loss for Person Re-Identification'."""
# 	def __init__(self, margin=None, metric="euclidean"):
# 		self.margin = margin
# 		self.metric = metric
# 		if margin is not None:
# 			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
# 		else:
# 			self.ranking_loss = nn.SoftMarginLoss()

# 	def __call__(self, global_feat, labels, normalize_feature=False):
# 		if normalize_feature:
# 			global_feat = normalize(global_feat, axis=-1)

# 		if self.metric == "euclidean":
# 			dist_mat = euclidean_dist(global_feat, global_feat)
# 		elif self.metric == "cosine":
# 			dist_mat = cosine_dist(global_feat, global_feat)
# 		else:
# 			raise NameError

# 		dist_ap, dist_an = hard_example_mining(
# 			dist_mat, labels)
# 		y = dist_an.new().resize_as_(dist_an).fill_(1)
		
# 		if self.margin is not None:
# 			loss = self.ranking_loss(dist_an, dist_ap, y)
# 		else:
# 			loss = self.ranking_loss(dist_an - dist_ap, y)
# 		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
# 		return loss