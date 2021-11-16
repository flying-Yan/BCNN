import torch
import torch.nn as nn
from models.binarized_fun import * 
import torchvision.transforms as transforms


class ResBlock_1(nn.Module):

    def __init__(self, inchannel):
        super(ResBlock_1, self).__init__()
        
        
        self.tanh = nn.Hardtanh(-1.3,1.3)
        
        self.conv1 = nn.Sequential(
                self.tanh,
                BinarizeConv2d(inchannel, inchannel, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(inchannel),
            )
    
    def forward(self, x):
            
        y = self.conv1(x)
        y = x + y

        return y
 
class ResBlock_2(nn.Module):

    def __init__(self, inchannel, outchannel):
        super(ResBlock_2, self).__init__()
        
        
        self.tanh = nn.Hardtanh(-1.3,1.3)
        
        self.conv1 = nn.Sequential(
                self.tanh,
                BinarizeConv2d(inchannel, outchannel, kernel_size = 3, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(outchannel),
            )
        self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(inchannel, outchannel, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
            
        y = self.conv1(x)
        x = self.shortcut(x)
        y = x + y

        return y
       
class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        
        self.tanh = nn.Hardtanh(-1.3,1.3,inplace=True)

        self.conv1 = nn.Sequential(
                nn.BatchNorm2d(3, affine = False),
                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
            )

        
        self.block_1_1 = ResBlock_1(64)
        self.block_1_2 = ResBlock_1(64)
        self.block_1_3 = ResBlock_1(64)
        self.block_1_4 = ResBlock_1(64)
        
        self.block_2_1 = ResBlock_2(64,128)
        self.block_2_2 = ResBlock_1(128)
        self.block_2_3 = ResBlock_1(128)
        self.block_2_4 = ResBlock_1(128)

        self.block_3_1 = ResBlock_2(128,256)
        self.block_3_2 = ResBlock_1(256)
        self.block_3_3 = ResBlock_1(256)
        self.block_3_4 = ResBlock_1(256)

        self.block_4_1 = ResBlock_2(256,512)
        self.block_4_2 = ResBlock_1(512)
        self.block_4_3 = ResBlock_1(512)
        self.block_4_4 = ResBlock_1(512)

        self.fc = nn.Sequential(
                nn.Linear(512,1000)
            )

        self.avg = nn.Sequential(
                nn.ReLU(),
                nn.AvgPool2d(7,7),
            )
        
        
        self.regime = {
            0: {'optimizer': 'Adam','betas': (0.9, 0.999), 'lr': 5e-3},
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
        
        x = self.conv1(x)   
        
        x = self.block_1_1(x)
        x = self.block_1_2(x)
        x = self.block_1_3(x)
        x = self.block_1_4(x)

        x = self.block_2_1(x)
        x = self.block_2_2(x)
        x = self.block_2_3(x)
        x = self.block_2_4(x)

        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.block_3_3(x)
        x = self.block_3_4(x)

        x = self.block_4_1(x)
        x = self.block_4_2(x)
        x = self.block_4_3(x)
        x = self.block_4_4(x)

        x = self.avg(x)

        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x



