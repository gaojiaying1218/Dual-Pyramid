from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import random
import math
import os
import re
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
img_path2 = r'D:\DataSet\RCD\JiDa\10017\bz_01_1_32_1_6.jpg'
img_path = r'D:\Paper\论文\我的论文\MM\图片2.jpg'
# img = Image.open(img_path)
# img = img.resize(224*224)




def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)






def add_noise_Guass(img, mean=0, var=0.01):  # 添加高斯噪声
    img = np.array(img / 255, dtype=float)# 将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    #0.01的0.5次幂,ctrl点击normal函数可见参数
    #给出均值为loc，标准差为scale的高斯随机数（场）
    '''
    numpy.random.normal(loc=0.0, scale=1.0, size=None)
    loc：float
    此概率分布的均值（对应着整个分布的中心centre）
    scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
    size：int or tuple of ints
    输出的shape，默认为None，只输出一个值
    '''
    out_img = img + noise# 将噪声和原始图像进行相加得到加噪后的图像
    if out_img.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
        out_img = np.clip(out_img, low_clip, 1.0)#clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        out_img = np.uint8(out_img * 255)# 解除归一化，乘以255将加噪后的图像的像素值恢复
    return out_img

img2 = cv2.imread(img_path2)  # 使用opencv获取的图片， 图片的类型为numpy.array
img = Image.open(img_path)
# img1 = np.array(img)
# out_img = add_noise_Guass(img2)
# cv2.imshow("imggaosi", out_img)
# cv2.waitKey(0)


d2l.set_figsize()
img = d2l.Image.open(img_path2)
d2l.plt.imshow(img);

color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))

augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)