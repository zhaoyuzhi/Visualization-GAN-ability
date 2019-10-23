# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 21:36:09 2018

@author: yzzhao2
"""

import argparse
import os
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.makedirs('images_VAE-GAN', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--num_workers', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=200, help='interval betwen image samples')
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument('--num_gmm', type=int, default=40, help='num of gmm')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
kwargs = {'num_workers': opt.num_workers, 'pin_memory': True} if cuda else {}

img_shape = (opt.channels, opt.img_size, opt.img_size)

# VAE structure
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # encoder
        self.fc1 = nn.Linear(784, 400)                                  # fc layer
        self.fc21 = nn.Linear(400, 40)                                  # mean - mu
        self.fc22 = nn.Linear(400, 40)                                  # variance - var
        # decoder
        self.fc3 = nn.Linear(40, 400)                                   # fc layer
        self.fc4 = nn.Linear(400, 784)                                  # fc layer

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterization(self, this_batch_size, mu, var):
        e = torch.randn(this_batch_size, 40)                            # e is standard normal distribution
        z = e.mul(var).add(mu)                                          # z is reparameterized normal distribution
        return z

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        return F.tanh(self.fc4(h3))

    def forward(self, x):
        this_batch_size = x.size(0)                                     # this_batch_size is not always equal to opt.batch_size
        mu, var = self.encoder(x)
        z = self.reparameterization(this_batch_size, mu, var)
        return self.decoder(z), mu, var, this_batch_size

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
# Reconstrcuct Loss
MSE_loss = torch.nn.MSELoss()
# KL Divergence Loss: 0.5 * sum(mu^2 + exp(var) - 1 - var)
def KL_loss(this_batch_size, mu, var):
    a = mu.pow(2).add(var.exp())
    b = torch.ones(this_batch_size, 40)
    b = b.add(var)
    out = 0.5 * (a - b)
    out = out.mean()
    return out
# GAN loss
adversarial_loss = nn.BCELoss()

# Initialize VAE
generator = Generator()
discriminator = Discriminator()

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
    batch_size = opt.batch_size, shuffle = True, **kwargs)
    
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Training

lambda_KL = 1
lambda_recon = 1

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

        optimizer_G.zero_grad()
        
        # Construct inputs
        imgs = imgs.view(-1, 784)
        
        # Generate a batch of images
        gen_imgs, mu, var, this_batch_size = generator(imgs)
    
        # Loss measures generator's ability to fool the discriminator, reconstruct and close to Normal Distribution
        reconstrcuct_loss = MSE_loss(gen_imgs, imgs)
        kl_loss = KL_loss(this_batch_size, mu, var)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        loss = lambda_recon * reconstrcuct_loss + lambda_KL * kl_loss + g_loss
        
        loss.backward()
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

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # if using VAE_DNN
            gen_imgs = gen_imgs.detach().resize_(gen_imgs.size(0), 1, 28, 28)
            
            save_image(gen_imgs.data[:25], 'images_VAE-GAN/%d.png' % batches_done, nrow=5, normalize=True)
