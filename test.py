#!/usr/bin/python3
# coding=utf-8
import os
import sys

sys.path.insert(0, '/')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

plt.ion()
import torch
import dataset
from torch.utils.data import DataLoader

from model.LPNet import LPNet
from tqdm import *




class Test(object):
    def __init__(self, Dataset, Network, Path, weight):

        ## dataset
        self.cfg = Dataset.Config(datapath=Path, snapshot=weight, mode='test')
        # self.cfg = Dataset.Config(datapath=Path, snapshot='', mode='test')

        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save_body_detail(self, save_name):
        res = []
        start_frame = 100
        count_num = 0
        with torch.no_grad():
            for image,depth, (H, W), name in tqdm(self.loader):
                count_num+=1

                image,depth, shape = image.cuda().float(), depth.cuda().float(),(H, W)
                torch.cuda.synchronize()
                start = time.time()
                out_trunk, out_struct, out_mask = self.net(image,depth)
                torch.cuda.synchronize()
                end = time.time()
                
                out_mask = F.interpolate(out_mask, size=shape, mode='bilinear')
                predmask = torch.sigmoid(out_mask[0, 0]).cpu().numpy() * 255
                
              
                head = save_name
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(predmask))
               

                if count_num>=start_frame:
                    res.append(end - start) 
                    time_sum = 0
                    for i in res:
                        time_sum += i
                    print("FPS: %f" % (1.0 / (time_sum / len(res))))



if __name__ == '__main__':
    # 测试集合
    current_directory = os.getcwd()
    # 获取当前文件夹的名字
    model_name = os.path.basename(current_directory)





    model_num = '48'
    test_set = 'DIS-VD'
    model_name = os.path.basename(current_directory)
    t = Test(dataset, LPNet, r'.../DIS5K/'+test_set, weight=r'./saveWeight/model-'+model_num)
    # 保存body和detail map
    t.save_body_detail(save_name=r'./results/'+model_num+'/'+test_set)
    
   