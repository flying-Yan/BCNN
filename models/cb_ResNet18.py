import torch
import torch.nn as nn
from models.binarized_fun import *  

class ResBlock_1(nn.Module):

    def __init__(self, inchannel):
        super(ResBlock_1, self).__init__()
        
        self.tanh = nn.Hardtanh(-1.3,1.3)
        
        self.conv1 = nn.Sequential(
                self.tanh,
                BinarizeConv2d(inchannel, inchannel, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(inchannel),
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
                self.tanh,
                BinarizeConv2d(outchannel, outchannel, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(outchannel),
            )
        self.shortcut = nn.Sequential(
                self.tanh,
                BinarizeConv2d(inchannel, outchannel, kernel_size = 1, stride = 2, padding = 0, bias = False),
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
                nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(64),
            )

        
        self.block_1_1 = ResBlock_1(64)
        self.block_1_2 = ResBlock_1(64)
        
        self.block_2_1 = ResBlock_2(64,128)
        self.block_2_2 = ResBlock_1(128)

        self.block_3_1 = ResBlock_2(128,256)
        self.block_3_2 = ResBlock_1(256)

        self.block_4_1 = ResBlock_2(256,512)
        self.block_4_2 = ResBlock_1(512)

        self.fc = nn.Sequential(
                nn.Linear(512,10)
            )

        self.avg = nn.Sequential(
                nn.ReLU(),
                nn.AvgPool2d(4,4),
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
        
        x = self.conv1(x)   
        
        x = self.block_1_1(x)
        x = self.block_1_2(x)

        x = self.block_2_1(x)
        x = self.block_2_2(x)

        x = self.block_3_1(x)
        x = self.block_3_2(x)

        x = self.block_4_1(x)
        x = self.block_4_2(x)

        x = self.avg(x)

        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x



