import numpy as np
import time
from vgg16 import VGG16
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

       

class FCN8(nn.Module):
    def __init__(self, num_class, weight_path=None):
        super().__init__()
        
        self.model = VGG16(True, weight_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc1 = nn.Conv2d(512, 4096, 1)
        self.fc2 = nn.Conv2d(4096, 4096, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()
        self.fc3 = nn.Conv2d(4096, num_class, 1)
        
        self.conv_t1 = nn.ConvTranspose2d(num_class, num_class, kernel_size = 4, stride=2, padding = 1)
        self.conv_t2 = nn.ConvTranspose2d(num_class, num_class, kernel_size = 4, stride=2, padding = 1)
        self.conv_t3 = nn.ConvTranspose2d(num_class, num_class, kernel_size = 16, stride=8, padding = 4)
        
        self.score_1 = nn.Conv2d(512, num_class, 1)
        self.score_2 = nn.Conv2d(256, num_class, 1)
        
    def forward(self, x):
        out, (layer3, layer4) = self.model(x)
        out = self.drop1(self.relu1(self.fc1(out)))
        out = self.drop2(self.relu2(self.fc2(out)))
        out = self.fc3(out)
        
        layer4 = self.score_1(layer4)
        layer3 = self.score_2(layer3)
        
        out = self.conv_t1(out)
        out = out + layer4
        
        out = self.conv_t2(out)
        out = out + layer3
        
        out = self.conv_t3(out)
        
        return out
        
    def extract_parameters(self):
        param = list(self.fc1.parameters())
        param += list(self.fc2.parameters())
        param += list(self.fc3.parameters())
        param += list(self.conv_t1.parameters())
        param += list(self.conv_t2.parameters())
        param += list(self.conv_t3.parameters())
        param += list(self.score_1.parameters())
        param += list(self.score_2.parameters())
        
        return param
