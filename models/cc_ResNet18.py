import torch
import torch.nn as nn
from models.binCom_fun import *

class ResBlock_1(nn.Module):

    def __init__(self, inchannel):
        super(ResBlock_1, self).__init__()
        
        
        self.tanh = nn.Hardtanh(-1.3,1.3)
        
        self.conv1 = nn.Sequential(
                self.tanh,
                BC_conv(inchannel, inchannel, kernel_size = [3,3], stride = 1, padding = 1, first_layer = False, bias = False),
                DC_Bn4(inchannel),
                self.tanh,
                BC_conv(inchannel, inchannel, kernel_size = [3,3], stride = 1, padding = 1, first_layer = False, bias = False),
                DC_Bn4(inchannel),
            )
    
    def forward(self, x):
            
        y = self.conv1(x)
        y = x + y

        return y
 
class ResBlock_2(nn.Module):

    def __init__(self, inchannel,outchannel):
        super(ResBlock_2, self).__init__()
        
        
        self.tanh = nn.Hardtanh(-1.3,1.3)
        
        self.conv1 = nn.Sequential(
                self.tanh,
                BC_conv(inchannel, outchannel, kernel_size = [3,3], stride = 2, padding = 1, first_layer = False, bias = False),
                DC_Bn4(outchannel),
                self.tanh,
                BC_conv(outchannel, outchannel, kernel_size = [3,3], stride = 1, padding = 1, first_layer = False, bias = False),
                DC_Bn4(outchannel),
            )
        self.shortcut = nn.Sequential(
                self.tanh,
                BC_conv(inchannel, outchannel, kernel_size = [1,1], stride = 2, padding = 0, first_layer = False, bias = False),
                DC_Bn4(outchannel),

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
        
        self.inputA = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size = 1, bias = False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size = 1, bias = False)
            )

                
        self.conv1 = nn.Sequential(
                DC_Bn4(3, scale = False),
                BC_conv(3, 46, kernel_size = [3,3], stride = 1, padding = 1, first_layer = True, bias = False),
                DC_Bn4(46),

            )

        
        self.block_1_1 = ResBlock_1(46)
        self.block_1_2 = ResBlock_1(46)
        
        self.block_2_1 = ResBlock_2(46,92)
        self.block_2_2 = ResBlock_1(92)

        self.block_3_1 = ResBlock_2(92,184)
        self.block_3_2 = ResBlock_1(184)

        self.block_4_1 = ResBlock_2(184,356)
        self.block_4_2 = ResBlock_1(356)

        self.fc = nn.Sequential(
                nn.Linear(356*2,10)
            )

        self.avg = nn.Sequential(
                nn.ReLU(),
                nn.AvgPool2d(4,4)
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
        
        y = self.inputA(x)
        x = torch.cat((x,y), dim = 1)

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



