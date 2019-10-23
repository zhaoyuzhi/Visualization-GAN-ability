f
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 21:36:09 2018

@author: yzzhao2
"""

import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images_GAN', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--pre_epochs', type=int, default=50, help='number of epochs of pre_training')
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_pre', type=float, default=0.0002, help='SGD: learning rate')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=200, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),       # 100 latent variables as input
            *block(128, 256),                                   # *list代表取出list中的每个元素
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),           # np.prod 连乘: np.prod(img_shape) = 1 * 28 * 28
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

# Loss function
adversarial_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
print(generator)
print(discriminator)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    MSE_loss.cuda()

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_pre = torch.optim.SGD(generator.parameters(), lr=opt.lr_pre)

Tensor = torch.cuda.FloatStorage if cuda else torch.FloatStorage

# --------------------
#     Pre-Training
# --------------------
'''
for epoch in range(opt.pre_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        optimizer_pre.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))   # PIL image imgs.shape[0]

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability
        pre_loss = MSE_loss(gen_imgs, imgs)

        pre_loss.backward()
        optimizer_pre.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [pre_training loss: %f] Pre-Training" % (epoch, opt.pre_epochs,
               i, len(dataloader), pre_loss.item()))
'''
# --------------------
#       Training
# --------------------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)   # PIL image imgs.size(0)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        for j in range(2):
            optimizer_G.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))   # PIL image imgs.shape[0]
    
            # Generate a batch of images
            gen_imgs = generator(z)
    
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    
            g_loss.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        if (i % 10) == 0:
            print(g_loss.item())
            print(d_loss.item())

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images_GAN/%d.png' % batches_done, nrow=5, normalize=True)
