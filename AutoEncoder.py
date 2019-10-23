# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 13:43:46 2018

@author: yzzhao2
"""

import argparse
import os
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F


os.makedirs('images_AutoEncoder', exist_ok=True)

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
        # encoder embedding is 20
        self.fc1 = nn.Linear(784, 200)                                  # fc layer
        self.fc2 = nn.Linear(200, 100)                                  # fc layer
        self.code1 = nn.Linear(100, 20)                                 # fc layer
        self.code2 = nn.Linear(20, 100)                                 # fc layer
        self.fc3 = nn.Linear(100, 200)                                  # fc layer
        self.fc4 = nn.Linear(200, 784)                                  # fc layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        code = F.relu(self.code1(x))
        x = F.relu(self.code2(code))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return code, x

# Loss function
MSE_loss = torch.nn.MSELoss()

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
             
        # Resize
        imgs = imgs.view(-1, 784)
        
        # Training
        optimizer.zero_grad()
        
        # Generate a batch of images
        codes, gen_imgs = model(imgs)

        # Loss measures
        loss = MSE_loss(gen_imgs, imgs)
        loss.backward()
        optimizer.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader), loss.item()))
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:

            # if using AutoEncoder
            gen_imgs = gen_imgs.detach().resize_(gen_imgs.size(0), 1, 28, 28)
            
            save_image(gen_imgs.data[:25], 'images_AutoEncoder/%d.png' % batches_done, nrow=5, normalize=True)
