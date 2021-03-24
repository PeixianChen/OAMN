from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch
import numpy as np

from PIL import Image
from torchvision.transforms import Resize
import torch
import math
import random

from . import transforms as T

class IdentityPreprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(IdentityPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.pindex = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, domainall = self.dataset[index]
        fpath = fname
        try:
            if self.root is not None:
                fpath = osp.join(self.root, fname)
            img = Image.open(fpath).convert('RGB')
        except:
            fpath = osp.join(self.root_, fname)
            img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, domainall

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img, fname, pid, camid,domainall = self.dataset[index]
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # return img, fname, pid, camid,domainall
        return img, fname, pid, camid, index

class Preprocessor_occluded(object):
    def __init__(self, dataset, root=None, transform=None, train=False):
        super(Preprocessor_occluded, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.train = train
        self.img_pil_crop = Image.new('RGB', (256, 256), (0, 0, 0))
        self.idx = [i for i in range(len(self.dataset))]
        # random.shuffle(self.idx)

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        self.add_transform = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.Pad(10),
            T.RandomCrop((256,128)),
            T.RandomRotation(5), 
            T.ToTensor(),
            normalizer,
        ])
        self.ori_transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
        ])

    def get_params(self, img0, scale=(0.01, 0.03), ratio=(0.3, 3.3)): #随机my2
        img = img0.copy()
        img_array = np.array(img).transpose(2, 0, 1)  # transpose (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array)
        img_c, img_h, img_w = img_tensor.shape
        area = img_h * img_w

        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                # 只获取4个角
                i = random.sample([0,img_h-h], 1)[0]
                j = random.sample([0,img_w-w], 1)[0]

        return i, j, h, w

    def occluded(self, img_pil0,img_pil_crop, position, size0):
        img_pil=img_pil0.copy()
        resize = Resize(
        size=(size0[0], size0[1]),  # (height, width)
        )
        img_pil_crop=resize(img_pil_crop)
        img_pil.paste(img_pil_crop,(position[1],position[0],position[1] + size0[1],position[0] + size0[0]))
        return img_pil

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img, fname, pid, camid, domainall = self.dataset[index]
        img = img.convert('RGB')
        # add image:
        img_pil= Image.open(fname)
        resize = Resize(
        size=(256, 128),  # (height, width)
        )
        img_pil=resize(img_pil)

        # 随机一张图片作为贴图
        img_pil_idx = Image.open(self.dataset[random.choice(self.idx)][1])
        img_pil_idx=resize(img_pil_idx)
        scale=(0.02, 0.2)
        ratio=(0.3, 3.3)
        i,j,h,w = self.get_params(img_pil_idx, scale, ratio)
        img_pil_crop=img_pil_idx.crop((j,i,j+w,i+h))#(l,u,r,d)

        size = random.sample([8192, 16384],1)[0]
        rl = size // 256
        position_l=[0,0]
        size_l = [256, rl]
        # 
        rr = size // 256
        position_r=[0,128-rr]
        size_r=[256,rr]
        # 
        ru = size // 128
        position_u=[0,0]
        size_u=[ru,128]
        # 
        rd = size // 128
        position_d=[256-rd,0]
        size_d=[rd,128]

        if self.train:
            img = self.ori_transform(img)
        
        if self.transform is not None:
            if self.train:
                img_u = self.occluded(img_pil, img_pil_crop, position_u, size_u)
                img_l = self.occluded(img_pil, img_pil_crop, position_l, size_l)
                img_r = self.occluded(img_pil, img_pil_crop, position_r, size_r)
                img_d = self.occluded(img_pil, img_pil_crop, position_d, size_d)

                # 
                img = self.add_transform(img).unsqueeze(0)
                img_l = self.add_transform(img_l).unsqueeze(0)
                img_r = self.add_transform(img_r).unsqueeze(0)
                img_u = self.transform(img_u).unsqueeze(0)
                img_d = self.transform(img_d).unsqueeze(0)
            else:
                # test:
                img_pil_crop = Image.new('RGB', (256, 256), (0, 0, 0))
                img_u = self.occluded(img_pil,img_pil_crop, [0,0], [128,128])
                img_l = self.occluded(img_pil,img_pil_crop, [0,0], [256,64])
                img_r = self.occluded(img_pil,img_pil_crop, [0,64], [256,64])
                img_d = self.occluded(img_pil,img_pil_crop, [128,0], [128,128])
                img_u_ = self.occluded(img_pil,img_pil_crop, [0,0], [64,128])
                img_l_ = self.occluded(img_pil,img_pil_crop, [0,0], [256,32])
                img_r_ = self.occluded(img_pil,img_pil_crop, [0,96], [256,32])
                img_d_ = self.occluded(img_pil,img_pil_crop, [192,0], [64,128])

                img = self.transform(img).unsqueeze(0)
                img_l = self.transform(img_l).unsqueeze(0)
                img_r = self.transform(img_r).unsqueeze(0)
                img_u = self.transform(img_u).unsqueeze(0)
                img_d = self.transform(img_d).unsqueeze(0)
                img_l_ = self.transform(img_l_).unsqueeze(0)
                img_r_ = self.transform(img_r_).unsqueeze(0)
                img_u_ = self.transform(img_u_).unsqueeze(0)
                img_d_ = self.transform(img_d_).unsqueeze(0)
                return torch.cat([img, img_l, img_r, img_d, img_u, img_l_, img_r_, img_d_, img_u_], 0), fname, pid, camid, index, size
        return torch.cat([img, img_l, img_r, img_d, img_u], 0), fname, pid, camid, index, size