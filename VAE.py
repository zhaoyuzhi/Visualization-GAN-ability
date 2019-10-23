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

os.makedirs('images_VAE', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--num_workers', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=200, help='interval betwen image samples')
parser.add_argument('--num_gmm', type=int, default=40, help='num of gmm')
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
kwargs = {'num_workers': opt.num_workers, 'pin_memory': True} if cuda else {}

# VAE structure
class VAE_DNN(nn.Module):
    def __init__(self):
        super(VAE_DNN, self).__init__()
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
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        this_batch_size = x.size(0)                                     # this_batch_size is not always equal to opt.batch_size
        mu, var = self.encoder(x)
        z = self.reparameterization(this_batch_size, mu, var)
        return self.decoder(z), mu, var, this_batch_size

'''
class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias='True')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias='True')
        self.fc11 = nn.Linear(320, 40)                                  # mean - mu
        self.fc12 = nn.Linear(320, 40)                                  # variance - var
        # decoder
        self.fc2 = nn.Linear(40, 320)                                   # fc layer
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size = 5, stride= 2, bias='True')
        self.deconv2 = nn.ConvTranspose2d(10, 10, kernel_size = 5, stride= 2, bias='True')
        self.deconv3 = nn.ConvTranspose2d(10, 1, kernel_size = 4, stride= 1, bias='True')

    def encoder(self, this_batch_size, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                      # 1*28*28 - 10*24*24 - 10*12*12
        x = F.relu(F.max_pool2d(self.conv2(x), 2))                      # 10*12*12 - 20*8*8 - 20*4*4
        x = x.view(this_batch_size, -1)
        mu = self.fc11(x)
        var = self.fc12(x)
        return mu, var

    def reparameterization(self, this_batch_size, mu, var):
        e = torch.randn(this_batch_size, 40)                            # e is standard normal distribution
        z = e.mul(var).add(mu)                                          # z is reparameterized normal distribution
        return z

    def decoder(self, this_batch_size, z):
        z = F.relu(self.fc2(z))
        z = z.reshape(this_batch_size, 20, 4, 4)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.sigmoid(self.deconv3(z))
        return z

    def forward(self, x):
        this_batch_size = x.size(0)                                     # this_batch_size is not always equal to opt.batch_size
        mu, var = self.encoder(this_batch_size, x)
        z = self.reparameterization(this_batch_size, mu, var)
        z = self.decoder(this_batch_size, z)
        return z, mu, var, this_batch_size
'''

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

# Initialize VAE
model = VAE_DNN()

if cuda:
    model.cuda()
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
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Training
for epoch in range(opt.n_epochs):
    model.train()
    
    for i, (imgs, _) in enumerate(dataloader):
        
        # To device
        if cuda:
            imgs.cuda()
             
        # Resize if use VAE_DNN
        imgs = imgs.view(-1, 784)
        
        # Training
        optimizer.zero_grad()
        
        # Generate a batch of images
        gen_imgs, mu, var, this_batch_size = model(imgs)

        # Loss measures
        reconstrcuct_loss = MSE_loss(gen_imgs, imgs)
        kl_loss = KL_loss(this_batch_size, mu, var)
        loss = reconstrcuct_loss + kl_loss

        loss.backward()
        optimizer.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [loss: %f] [MSE loss: %f] [KL loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            loss.item(), reconstrcuct_loss.item(), kl_loss.item()))
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # if using VAE_DNN
            gen_imgs = gen_imgs.detach().resize_(gen_imgs.size(0), 1, 28, 28)
            
            save_image(gen_imgs.data[:25], 'images_VAE/%d.png' % batches_done, nrow=5, normalize=True)
