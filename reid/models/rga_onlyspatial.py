# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F

import pdb

# ===================
#     RGA Module
# ===================

class RGA_Module(nn.Module):
	def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
		cha_ratio=8, spa_ratio=8, down_ratio=8):
		super(RGA_Module, self).__init__()

		self.in_channel = in_channel
		self.in_spatial = in_spatial
		
		self.use_spatial = use_spatial
		self.use_channel = use_channel

		print ('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

		self.inter_channel = in_channel // cha_ratio
		self.inter_spatial = in_spatial // spa_ratio
		
		# Embedding functions for original features
		if self.use_spatial:
			self.gx_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
		
		# Embedding functions for relation features
		if self.use_spatial:
			self.gg_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_spatial),
				nn.ReLU()
			)
		
		# Networks for learning attention weights
		if self.use_spatial:
			num_channel_s = 1 + self.inter_spatial
			self.W_spatial = nn.Sequential(
				nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s//down_ratio,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(num_channel_s//down_ratio),
				nn.ReLU(),
				nn.Conv2d(in_channels=num_channel_s//down_ratio, out_channels=1,
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(1)
			)

		# Embedding functions for modeling relations
		if self.use_spatial:
			self.theta_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
								kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)
			self.phi_spatial = nn.Sequential(
				nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
							kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.inter_channel),
				nn.ReLU()
			)

	def forward(self, x):
		b, c, h, w = x.size()
		
		if self.use_spatial:
			# spatial attention
			theta_xs = self.theta_spatial(x)	
			phi_xs = self.phi_spatial(x)
			theta_xs = theta_xs.view(b, self.inter_channel, -1)
			theta_xs = theta_xs.permute(0, 2, 1)
			phi_xs = phi_xs.view(b, self.inter_channel, -1)
			Gs = torch.matmul(theta_xs, phi_xs)
			Gs_in = Gs.permute(0, 2, 1).view(b, h*w, h, w)
			Gs_out = Gs.view(b, h*w, h, w)
			Gs_joint = torch.cat((Gs_in, Gs_out), 1)
			Gs_joint = self.gg_spatial(Gs_joint)
		
			g_xs = self.gx_spatial(x)
			g_xs = torch.mean(g_xs, dim=1, keepdim=True)
			ys = torch.cat((g_xs, Gs_joint), 1)

			W_ys = self.W_spatial(ys)
			out = torch.sigmoid(W_ys.expand_as(x))
			return out