import torch
import torch.nn as nn
from models.binCom_fun import *
import torchvision.transforms as transforms


class Net(nn.Module):

    def __init__(self, num_classes=1000):
        super(Net, self).__init__()
        
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
                nn.Hardtanh(-1.3,1.3,inplace = True),
            )
        
        def BC_after(inp, oup, kk, ss, pp):
            return nn.Sequential(
                BC_conv(inp, oup, kernel_size = [kk,kk], stride = ss, padding = pp, first_layer = False, bias = True),
                DC_Bn4(oup),
                nn.Hardtanh(-1.3,1.3,inplace = True),
            )
        
        def BC_pool(inp, oup, kk, ss, pp):
            return nn.Sequential(
                BC_conv(inp, oup, kernel_size = [kk,kk], stride = ss, padding = pp, first_layer = False, bias = True),
                BC_sPool(gamma = 0.5),
                DC_Bn4(oup),
                nn.Hardtanh(-1.3,1.3,inplace = True),
            )
        
        
        self.features = nn.Sequential(
            BC_first(3,62,11,4,0),
            BC_after(62,82,3,1,1),
            BC_pool(82,82,3,1,1),
            
            
            BC_after(82,164,5,1,2),
            BC_after(164,164,3,1,1),
            BC_pool(164,164,3,1,1),
            

            BC_after(164,246,3,1,1),
            BC_after(246,246,1,1,0),
            BC_pool(246,246,1,1,0),
            

            BC_after(246,656,3,1,1),
            BC_after(656,530,1,1,0),

                   
            nn.Conv2d(530*2, 1000, kernel_size=[1,1], padding = 0, bias=True),
            nn.BatchNorm2d(1000),

            nn.AvgPool2d(kernel_size=6, stride=6),

        
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
        
        y = self.inputA(x)
        x = torch.cat((x,y), dim = 1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        return x




