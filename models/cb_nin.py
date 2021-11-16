import torch
import torch.nn as nn
from models.binarized_fun import  *


# NIN-Net on Cifar10

class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.tanh = nn.Hardtanh(-1.3, 1.3, inplace=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            self.tanh,

            BinarizeConv2d(192, 160, kernel_size=1, padding=0),
            nn.BatchNorm2d(160),
            self.tanh,

            BinarizeConv2d(160, 96, kernel_size=1, padding=0),

            nn.MaxPool2d(kernel_size=2, stride=2, padding = 0),
            nn.BatchNorm2d(96),
            self.tanh,


            BinarizeConv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            self.tanh,

            BinarizeConv2d(192, 192, kernel_size=1, padding=0),
            nn.BatchNorm2d(192),
            self.tanh,

            BinarizeConv2d(192, 192, kernel_size=1, padding=0),

            nn.MaxPool2d(kernel_size=2, stride=2, padding = 0),
            nn.BatchNorm2d(192),
            self.tanh,

            BinarizeConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            self.tanh,

            BinarizeConv2d(192, 192, kernel_size=1, padding=0),
            

            nn.BatchNorm2d(192),
            self.tanh,
            
            nn.Conv2d(192, 10, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8, stride=1, padding = 0),

        )
               

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            80: {'lr': 1e-3},
            150: {'lr': 5e-4},
            200: {'lr': 1e-4},
            240: {'lr': 5e-5},
            270: {'lr': 1e-5}
        }

    def forward(self, x):
       
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        return x



