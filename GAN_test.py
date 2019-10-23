# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:55:07 2018

@author: yzzhao2
"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# 定义G
class generator(nn.Module):
    def __init__(self, input_size):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )
 
    def forward(self, x):
        x = self.gen(x)
        return x
G = generator()

# 定义D
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.dis(x)
        return x
D = discriminator()

# --------
# 进入D训练
# --------
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

img = img.view(num_img, -1)  # 将图片展开乘28x28=784
real_img = Variable(img).cuda()  # 将tensor变成Variable放入计算图中
real_label = Variable(torch.ones(num_img)).cuda()  # 定义真实label为1
fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的label为0
 
# compute loss of real_img
real_out = D(real_img)  # 将真实的图片放入判别器中
d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss  
real_scores = real_out  # 真实图片放入判别器输出越接近1越好
 
# compute loss of fake_img
z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成一些噪声
fake_img = G(z)  # 放入生成网络生成一张假的图片
fake_out = D(fake_img)  # 判别器判断假的图片
d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
fake_scores = fake_out  # 假的图片放入判别器越接近0越好
 
# bp and optimize
d_loss = d_loss_real + d_loss_fake  # 将真假图片的loss加起来
d_optimizer.zero_grad()  # 归0梯度
d_loss.backward()  # 反向传播
d_optimizer.step()  # 更新参数

# --------
# 进入G训练
# --------
# compute loss of fake_img
z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 得到随机噪声
fake_img = G(z)  # 生成假的图片
output = D(fake_img)  # 经过判别器得到结果
g_loss = criterion(output, real_label)  # 得到假的图片与真实图片label的loss
 
# bp and optimize
g_optimizer.zero_grad()  # 归0梯度
g_loss.backward()  # 反向传播
g_optimizer.step()  # 更新生成网络的参数
