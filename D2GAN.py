# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:12:43 2018

@author: ZHAO Yuzhi
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

os.makedirs('images_D2GAN', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=200, help='number of image channels')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(128*ds_size**2, 1),
            nn.Softplus()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(128*ds_size**2, 1),
            nn.Softplus()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Loss functions
class Log_loss(nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation = True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation = True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss

criterion_log = Log_loss()
criterion_itself = Itself_loss()

# Initialize generator and discriminator
generator = Generator()
discriminator1 = Discriminator1()
discriminator2 = Discriminator2()

if cuda:
    generator.cuda()
    discriminator1.cuda()
    discriminator2.cuda()
    criterion_log.cuda()
    criterion_itself.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator1.apply(weights_init_normal)
discriminator2.apply(weights_init_normal)

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()

        # train with true
        # D1 sees real as real, minimize -logD1(x)
        output = discriminator1(real_imgs)
        errD1_real = 0.2 * criterion_log(output)
        
        # D2 sees real as fake, minimize D2(x)
        output = discriminator2(real_imgs)
        errD2_real = criterion_itself(output, False)
        
        # train with fake
        gen_imgs = generator(z)
        
        # D1 sees fake as fake, minimize D1(G(z))
        output = discriminator1(gen_imgs.detach())
        errD1_fake = criterion_itself(output, False)
        
        # D2 sees fake as real, minimize -log(D2(G(z))
        output = discriminator2(gen_imgs.detach())
        errD2_fake = 0.1 * criterion_log(output)

        lossD1 = errD1_real + errD1_fake
        lossD2 = errD2_real + errD2_fake
        
        optimizer_D1.step()
        optimizer_D2.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # G: minimize -D1(G(z)): to make D1 see fake as real
        output = discriminator1(gen_imgs)
        errG1 = criterion_itself(output)
        
        # G: minimize logD2(G(z)): to make D2 see fake as fake
        output = discriminator2(gen_imgs)
        errG2 = criterion_log(output, False)
        
        lossG = errG2 * 0.1 + errG1
        lossG.backward()
        optimizer_G.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D1 loss: %f] [D2 loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            lossD1.item(), lossD2.item(), lossG.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images_D2GAN/%d.png' % batches_done, nrow=5, normalize=True)
