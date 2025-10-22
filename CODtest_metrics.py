# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang


import os
import cv2
from tqdm import tqdm
from sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure

# 直接指定 ground truth 和预测文件的路径

gt_path = ''  # 你的真实值文件夹
pred_path = ''  # 你的预测值文件夹
txtpath = ''  # 结果保存路径

# 初始化评估指标
FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
MAE = MAE()

# 获取文件列表，假设 gt 和 pred 中的文件名是一一对应的
gt_file_list = sorted(os.listdir(gt_path))
pred_file_list = sorted(os.listdir(pred_path))

# 确保 gt 和 pred 文件数量相同
assert len(gt_file_list) == len(pred_file_list), "ground truth 和 predictions 文件数量不一致！"

# 遍历所有文件，计算指标
for gt_file, pred_file in tqdm(zip(gt_file_list, pred_file_list), total=len(gt_file_list)):
    gt_file_fullpath = os.path.join(gt_path, gt_file)
    pred_file_fullpath = os.path.join(pred_path, pred_file)

    # 读取 gt 和 pred 图像
    mask = cv2.imread(gt_file_fullpath, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_file_fullpath, cv2.IMREAD_GRAYSCALE)

    # 确保图像大小一致
    if not mask.shape == pred.shape:
        rows, cols = mask.shape[:2]
        pred = cv2.resize(pred, (cols, rows), interpolation=cv2.INTER_CUBIC)

    # 更新指标
    FM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

# 计算结果
fm = FM.get_results()["fm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]

# 打印和保存结果
results = {
    "Smeasure": sm.round(3),
    "meanEm": em["curve"].mean().round(3),
    "meanFm": fm["curve"].mean().round(3),
    "MAE": mae.round(3),
}

print(results)

with open(txtpath, 'a+') as fp:
    fp.write((str(results).replace('{', '')).replace('}', ''))
    fp.write('\n')

print("程序运行完成！")



































