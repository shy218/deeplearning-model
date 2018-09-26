import numpy as np
import time

import torch
from torch import nn

import torch.nn.functional as F



class VGG16(nn.Module):
    def __init__(self, pre_trained = True, weight_path = None):
        super().__init__()
        
        self.layer1 = self.make_layer(2, 3, 64)
        self.layer2 = self.make_layer(2, 64, 128)
        self.layer3 = self.make_layer(3, 128, 256)
        self.layer4 = self.make_layer(3, 256, 512)
        self.layer5 = self.make_layer(3, 512, 512)
        self. weight_path = weight_path
        if pre_trained:
            self.init_weight()
        
    def init_weight(self):
        
        assert self.weight_path, 'Need to download weight from "https://download.pytorch.org/models/vgg16-397923af.pth"'
        w = torch.load(self.weight_path)
        
        count = 0
        for i in range(2):
            self.layer1[i*2].weight.data = w['features.{}.weight'.format(i*2+count)]
            self.layer1[i*2].bias.data = w['features.{}.bias'.format(i*2+count)]
        count += 5
        
        for i in range(2):
            self.layer2[i*2].weight.data = w['features.{}.weight'.format(i*2+count)]
            self.layer2[i*2].bias.data = w['features.{}.bias'.format(i*2+count)]
        count += 5
        
        for i in range(3):
            self.layer3[i*2].weight.data = w['features.{}.weight'.format(i*2+count)]
            self.layer3[i*2].bias.data = w['features.{}.bias'.format(i*2+count)]
        count += 7
        
        for i in range(3):
            self.layer4[i*2].weight.data = w['features.{}.weight'.format(i*2+count)]
            self.layer4[i*2].bias.data = w['features.{}.bias'.format(i*2+count)]
        count += 7
        
        for i in range(3):
            self.layer5[i*2].weight.data = w['features.{}.weight'.format(i*2+count)]
            self.layer5[i*2].bias.data = w['features.{}.bias'.format(i*2+count)]
        
    
    
    def make_layer(self, num_block, inplane, outplane):
        layer = []
        layer.append(nn.Conv2d(inplane, outplane, kernel_size=3, stride=1, padding=1))
        layer.append(nn.ReLU(inplace=True))
        for i in range(1, num_block):
            layer.append(nn.Conv2d(outplane, outplane, kernel_size=3, stride=1, padding=1))
            layer.append(nn.ReLU(inplace=True))
        layer.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        residual = []
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        residual.append(x)
        x = self.layer4(x)
        residual.append(x)
        x = self.layer5(x)
        
        return x, residual

