#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset




class Normalize(object):
    def __init__(self, mean, std, depth_mean=124.55, depth_std=56.77):
        
        self.mean = mean
        self.std = std
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(self, image, mask=None, trunk=None, struct=None, depth=None):
        image = (image - self.mean) / self.std

        if mask is None:
            if depth is not None:
                depth = (depth - self.depth_mean) / self.depth_std
                return image, depth
            return image

        mask = mask / 255.0
        trunk = trunk / 255.0
        struct = struct / 255.0
    
        if depth is not None:
            depth = (depth - self.depth_mean) / self.depth_std

        return image, mask, trunk, struct, depth


class RandomCrop(object):
    def __call__(self, image, mask=None, trunk=None, struct=None, depth=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], trunk[p0:p1, p2:p3], struct[p0:p1, p2:p3], depth[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None, trunk=None, struct=None, depth=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return (image[:, ::-1, :].copy(), mask[:, ::-1].copy(), trunk[:, ::-1].copy(), 
                    struct[:, ::-1].copy(), depth[:, ::-1].copy())
        else:
            if mask is None:
                return image
            return image, mask, trunk, struct, depth



class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, trunk=None, struct=None, depth=None):
        
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        
        
        if mask is None:
            depth = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, depth
     
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        trunk = cv2.resize(trunk, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        struct = cv2.resize(struct, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        
        return image, mask, trunk, struct, depth



class ToTensor(object):
    def __call__(self, image, mask=None, trunk=None, struct=None, depth=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)  # 调整通道顺序为 (C, H, W)
        
        if mask is None:
            if depth is not None:
                depth = torch.from_numpy(depth)  # 转为 Tensor
            return image, depth
        

        mask = torch.from_numpy(mask)
        trunk = torch.from_numpy(trunk)
        struct = torch.from_numpy(struct)
        depth = torch.from_numpy(depth)
        return image, mask, trunk, struct, depth


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(1024, 1024)
        self.totensor = ToTensor()

        self.samples = os.listdir(os.path.join(self.cfg.datapath, 'Imgs'))

    def __getitem__(self, idx):
        name = self.samples[idx]
        name = name[0:-4]

        image = cv2.imread(self.cfg.datapath + '/Imgs/' + name + '.jpg')[:, :, ::-1].astype(np.float32)
        depth = cv2.imread(self.cfg.datapath + '/Depth/' + name + '.png', 0).astype(np.float32)

        if self.cfg.mode == 'train':
            mask = cv2.imread(self.cfg.datapath + '/GT/' + name + '.png', 0).astype(np.float32)
            trunk = cv2.imread(self.cfg.datapath + '/trunk-origin/' + name + '.png', 0).astype(np.float32)
            struct = cv2.imread(self.cfg.datapath + '/struct-origin/' + name + '.png', 0).astype(np.float32)

            image, mask, trunk, struct, depth = self.normalize(image, mask, trunk, struct, depth)
            image, mask, trunk, struct, depth = self.randomcrop(image, mask, trunk, struct, depth)
            image, mask, trunk, struct, depth = self.randomflip(image, mask, trunk, struct, depth)
            return image, mask, trunk, struct, depth
        else:
            shape = image.shape[:2]
            image, depth = self.normalize(image, depth=depth)
            image, depth = self.resize(image, depth=depth)
            image, depth = self.totensor(image, depth=depth)
            return image, depth, shape, name


    
    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        size = [1024, 1024, 1024, 1024, 1024][np.random.randint(0, 5)]
        image, mask, trunk, struct, depth = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            trunk[i] = cv2.resize(trunk[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            struct[i] = cv2.resize(struct[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            depth[i] = cv2.resize(depth[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        trunk = torch.from_numpy(np.stack(trunk, axis=0)).unsqueeze(1)
        struct = torch.from_numpy(np.stack(struct, axis=0)).unsqueeze(1)
        depth = torch.from_numpy(np.stack(depth, axis=0)).unsqueeze(1)

        return image, mask, trunk, struct, depth
















