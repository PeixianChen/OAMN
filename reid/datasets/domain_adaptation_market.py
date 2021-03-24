from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import random
import pdb
from glob import glob
import re

from PIL import Image
from torchvision.transforms import Resize
import torch
import math


class DA(object):

    def __init__(self, data_dir, source, target):

        self.source_images_dir = osp.join(data_dir, source)
        target_1 = "market1501_"
        target_2 = "Partial_iLIDS"
        target_3 = "Partial-REID_Dataset"
        self.target_images_dir_1 = osp.join(data_dir, target_1)
        self.target_images_dir_2 = osp.join(data_dir, target_2)
        self.target_images_dir_3 = osp.join(data_dir, target_3)

        self.source_train_path = 'bounding_box_train'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'

        self.source_train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_generate_ids = 0, 0, 0, 0
        self.source_num_cam = 6 if 'market' in source else 8
        self.pid_num = 0
        self.load()
    
    def preprocess(self, images_dir, path, relabel=True):
        self.pid_num = 0
        pattern = re.compile(r'([-\d]+)_c?(\d+)(\w+)(?:-(\d+))?\.jpg')

        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        if fpaths == []:
            fpaths = sorted(glob(osp.join(images_dir, path, '*.bmp')))
        fpaths = sorted(fpaths)
        for fpath in fpaths:
            fname = fpath
            pid, cam, _, pindex = map(str, pattern.search(fname).groups())
            cam = int(cam)
            pid = int(pid)
            if not (pindex == 'None'): 
                pindex = int(pindex)
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = self.pid_num
                    self.pid_num += 1
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            if pid > self.pid_num: self.pid_num = pid
            cam -= 1
            
            # add image:
            img_pil= Image.open(fname)
            resize = Resize(
            size=(256, 128),  # (height, width)
            )
            img_pil=resize(img_pil)
            ret.append((img_pil, fname, pid, cam, pindex))

        return ret, int(len(all_pids)), len(ret)

    def load(self):
        self.source_train, self.num_train_ids, self.source_pindex = self.preprocess(self.source_images_dir, self.source_train_path)
        self.gallery_2, self.num_gallery_ids_2, _ = self.preprocess(self.target_images_dir_2, self.gallery_path, False)
        self.query_2, self.num_query_ids_2, _ = self.preprocess(self.target_images_dir_2, self.query_path, False)

        self.gallery_3, self.num_gallery_ids_3, _ = self.preprocess(self.target_images_dir_3, self.gallery_path, False)
        self.query_3, self.num_query_ids_3, _ = self.preprocess(self.target_images_dir_3, self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset          |  # ids  | # images")
        print("  ------------------------------------")
        print("  source train    |  {:5d}  | {:8d}"
              .format(self.num_train_ids, len(self.source_train)))
        print("  query_partialiLIDS           |  {:5d}  | {:8d}"
              .format(self.num_query_ids_2, len(self.query_2)))
        print("  gallery_partialiLIDS         |  {:5d}  | {:8d}"
              .format(self.num_gallery_ids_2, len(self.gallery_2)))
        print("  query_partralREID          |  {:5d}  | {:8d}"
              .format(self.num_query_ids_3, len(self.query_3)))
        print("  gallery_partralREID        |  {:5d}  | {:8d}"
              .format(self.num_gallery_ids_3, len(self.gallery_3)))