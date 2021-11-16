import torch
import torch.nn as nn
from models.binCom_fun import *


class Net(nn.Module):

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        
        self.tanh = nn.Hardtanh(-1.3,1.3,inplace = True)
        self.inputA = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size = 1, bias = False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size = 1, bias = False)
            )
        
        def BC_first(inp, oup, kk, ss, pp):
            return nn.Sequential(
                BC_conv(inp, oup, kernel_size = [kk,kk], stride = ss, padding = pp, first_layer = True, bias = True),
                DC_Bn4(oup),
                self.tanh
            )
        def BC_after(inp, oup, kk, ss, pp):
            return nn.Sequential(
                BC_conv(inp, oup, kernel_size = [kk,kk], stride = ss, padding = pp, first_layer = False, bias = True),
                DC_Bn4(oup),
                self.tanh
            )
        def BC_pool(inp, oup, kk, ss, pp):
            return nn.Sequential(
                BC_conv(inp, oup, kernel_size = [kk,kk], stride = ss, padding = pp, first_layer = False, bias = True),
                BC_sPool(gamma = 0.5),
                DC_Bn4(oup),
                self.tanh
            )


        self.features = nn.Sequential(
            BC_first(3,124,5,1,2),
            BC_after(124,100,1,1,0),
            BC_pool(100,60,1,1,0),
            BC_after(60,124,5,1,2),
            BC_after(124,124,1,1,0),
            BC_pool(124,124,1,1,0),
            BC_after(124,124,3,1,1),
            
            BC_conv(124,124 , kernel_size=[1,1], padding=0, bias=True),
            DC_Bn4(124),
            nn.Hardtanh(inplace = True),
            
            nn.Conv2d(124*2,  10, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(10),
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
        y = self.inputA(x)
        x = torch.cat((x,y), dim = 1)

        x = self.features(x)
        x = x.view(-1, 10)

        return x




