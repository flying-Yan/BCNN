import torch
import torch.nn as nn
from models.binarized_fun import * 
import torchvision.transforms as transforms


class Net(nn.Module):

    def __init__(self, num_classes=1000):
        super(Net, self).__init__()
        
        self.tanh = nn.Hardtanh(-1.3, 1.3, inplace=True)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            self.tanh,

            BinarizeConv2d(96, 128, kernel_size=3, padding = 1),
            nn.BatchNorm2d(128),
            self.tanh,

            BinarizeConv2d(128, 128, kernel_size=3, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            self.tanh,

            
            BinarizeConv2d(128,256,kernel_size=5,padding=2),
            nn.BatchNorm2d(256),
            self.tanh,

            BinarizeConv2d(256, 256, kernel_size=3,padding = 1 ),
            nn.BatchNorm2d(256),
            self.tanh,

            BinarizeConv2d(256, 256, kernel_size=3, padding = 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            self.tanh,

            
            BinarizeConv2d(256,384,kernel_size=3,padding=1),
            nn.BatchNorm2d(384),
            self.tanh,
            
            BinarizeConv2d(384,384,kernel_size=1),
            nn.BatchNorm2d(384),
            self.tanh,

            BinarizeConv2d(384,384,kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(384),
            self.tanh,

            
            BinarizeConv2d(384,1024,kernel_size=3,padding=1),
            nn.BatchNorm2d(1024),
            self.tanh,
            
            BinarizeConv2d(1024,1024,kernel_size=1),
            nn.BatchNorm2d(1024),
            self.tanh,

            nn.Conv2d(1024,1000,kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=6, stride=1),
        )
        
        
        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            25: {'lr': 1e-3},
            35: {'lr': 5e-4},
            40: {'lr': 1e-4},
            45: {'lr': 1e-5}
        }
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.input_transform = {
            'train': transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'eval': transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        }

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        return x




