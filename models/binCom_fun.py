import torch
import torch.nn as nn
import math
import numpy as np
#from .binarized_modules import  BinarizeLinear,BinarizeConv2d,Binarize
import torch.nn.functional as F
#from .input_8b import *

def binary(x):
# bianrized tensor by using sign
    return torch.sign(x + 1e-6)





class BC_conv(nn.Module):
# input--N C_in H W
# output-- N C_out H W
# weight-- outchannel, in_channel, kernel_size[0], kernel_size[1]
    def __init__(self, in_channel, out_channel, kernel_size = [3,3],stride = [1,1],padding = 1 ,dilation = 1,bias = True,first_layer = False,group = 1):
        super(BC_conv,self).__init__()
        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
        self.first_layer = first_layer
        self.group = group
        
        #self.init = init

        self.weight_real = nn.Parameter(torch.Tensor(self.out_channel,self.in_channel,*kernel_size))
        self.weight_imag = nn.Parameter(torch.Tensor(self.out_channel,self.in_channel,*kernel_size))
        
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channel*2))
        else:
            self.register_parameter('bias',None)
    
        fanIO = nn.init._calculate_fan_in_and_fan_out(self.weight_real)        
        self.fan_in = fanIO[0]
        self.fan_out = fanIO[1]
        
        #self.reset_parameters_003()
        
        self.reset_parameters()
               #if self.init == 11:
            #self.reset_parameters_011()
        #if self.init == 12:
            #self.reset_parameters_012()

          
        
    def reset_parameters(self):      #conv2d xavier glorot try normal
        nn.init.xavier_uniform_(self.weight_real, gain = 4.)
        nn.init.xavier_uniform_(self.weight_imag, gain = 4.)
        #nn.init.xavier_normal_(self.weight_real)
        #nn.init.xavier_normal_(self.weight_imag)

        if self.bias is not None:
            bound = 1.0/math.sqrt(self.fan_in + self.fan_out)
            nn.init.uniform_(self.bias,-bound,bound)
            #nn.init.zeros_(self.bias)
    
        
    
    def forward(self, input):

        if self.first_layer is False:
            input.data = binary(input.data)

        if not hasattr(self.weight_real,'org'):
            self.weight_real.org = self.weight_real.data.clone()
            self.weight_imag.org = self.weight_imag.data.clone()
        self.weight_real.data = binary(self.weight_real.org)
        self.weight_imag.data = binary(self.weight_imag.org)


        kernel_real = torch.cat((self.weight_real,-self.weight_imag),dim = 1)
        kernel_imag = torch.cat((self.weight_imag, self.weight_real),dim = 1)
        kernel_complex = torch.cat((kernel_real,kernel_imag),dim = 0)
        
        #out = nn.functional.conv2d(input,kernel_complex,self.bias,self.stride,self.padding,self.dilation,self.group)
        out = nn.functional.conv2d(input,kernel_complex,None,self.stride,self.padding,self.dilation,self.group)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1,-1,1,1).expand_as(out)

        return out






class BC_fc(nn.Module):
# bianry complex fully connected layer  input[batch,norm]
    def __init__(self, fan_in, fan_out, bias = False,first_layer = False):
        super(BC_fc, self).__init__()
        self.fan_in = int(fan_in)
        self.fan_out = int(fan_out)
        self.first_layer = first_layer
        self.weight_real = nn.Parameter(torch.Tensor(self.fan_out,self.fan_in))
        self.weight_imag = nn.Parameter(torch.Tensor(self.fan_out,self.fan_in))
        
        #self.init = init
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.fan_out*2))
        else:
            self.register_parameter('bias', None)
        
        #if self.init == 2:
            #self.reset_parameters_00()
        #else:
            #if self.init == 3:
                #self.reset_parameters_01()
            #else:
                #self.reset_parameters_2()
        
        self.reset_parameters()


    def reset_parameters(self):      #conv2d xavier glorot try normal
        nn.init.xavier_uniform_(self.weight_real, gain = 4.)
        nn.init.xavier_uniform_(self.weight_imag, gain = 4.)
        #nn.init.xavier_normal_(self.weight_real)
        #nn.init.xavier_normal_(self.weight_imag)

        if self.bias is not None:
            bound = 1.0/math.sqrt(self.fan_in + self.fan_out)
            nn.init.uniform_(self.bias,-bound,bound)
            #nn.init.zeros_(self.bias)

            #nn.init.zeros_(self.bias)


    def forward(self, input):
        
        if self.first_layer is False:
            input.data = binary(input.data)

        if not hasattr(self.weight_real,'org'):
            self.weight_real.org = self.weight_real.data.clone()
            self.weight_imag.org = self.weight_imag.data.clone()

        self.weight_real.data = binary(self.weight_real.org)
        self.weight_imag.data = binary(self.weight_imag.org)
        
        
        kernel_real = torch.cat((self.weight_real,-self.weight_imag),dim = -1)
        kernel_imag = torch.cat((self.weight_imag, self.weight_real),dim = -1)
        kernel_complex = torch.cat((kernel_real,kernel_imag),dim = 0) 

        #out = nn.functional.linear(input,kernel_complex,self.bias)
        out = nn.functional.linear(input,kernel_complex)
        
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1,-1).expand_as(out)

        return out

        



class BC_sPool(nn.Module):
    def __init__(self,gamma = 0.5,norm = True):
        super(BC_sPool,self).__init__()
        self.gamma = gamma
        self.norm = norm

    def forward(self,input):
        xshape = input.shape
        x_per = input.permute(0,2,3,1)
        x_dim = int(xshape[1]/2)
        x_real = input[:,:x_dim,:,:]
        x_imag = input[:,x_dim:,:,:]
       
        xx = torch.cat((x_real[:,:,:,:,None],x_imag[:,:,:,:,None]),dim = 4)
        
        xff = torch.fft(xx,2,self.norm)
       
        topf1 = int(math.ceil(xshape[2]*self.gamma/2))
        topf2 = int(math.ceil(xshape[3]*self.gamma/2))
        midf1 = int(xshape[2]/2) + topf1
        midf2 = int(xshape[3]/2) + topf2
       
        f11 = xff[:,:,:topf1,:topf2,:]
        f12 = xff[:,:,midf1:,:topf2,:]
        f21 = xff[:,:,:topf1,midf2:,:]
        f22 = xff[:,:,midf1:,midf2:,:]
        
        fff1 = torch.cat((f11,f12),dim = 2)
        fff2 = torch.cat((f21,f22),dim = 2)

        f_pool = torch.cat((fff1,fff2),dim = 3)
        
        ifft = torch.ifft(f_pool,2,self.norm)
        
        out = torch.cat((ifft[:,:,:,:,0],ifft[:,:,:,:,1]),dim = 1)
        
        return out


class DC_Bn4(nn.Module):
    def __init__(self, num_feature, axis = -1, momentum = 0.9, eps = 1e-5, scale = True):
        super(DC_Bn4, self).__init__()
        self.num_feature = num_feature
        self.axis = axis
        self.momentum = momentum
        self.eps = eps
        self.scale = scale
        
        self.register_buffer('moving_mean', torch.zeros(num_feature*2))
        self.register_buffer('moving_rr', torch.ones(num_feature*2))
        self.register_buffer('moving_ri', torch.ones(num_feature))
        self.register_buffer('moving_ii', torch.ones(num_feature))

        if scale:
            self.gamma_rr = nn.Parameter(torch.Tensor(num_feature))
            self.gamma_ri = nn.Parameter(torch.Tensor(num_feature))
            self.gamma_ii = nn.Parameter(torch.Tensor(num_feature))
            self.beta = nn.Parameter(torch.Tensor(num_feature*2))
        else:
            self.register_parameter('gamma_rr',None)
            self.register_parameter('gamma_ri',None)
            self.register_parameter('gamma_ii',None)
            self.register_parameter('beta',None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.moving_mean, 0)
        nn.init.constant_(self.moving_rr, 1/math.sqrt(2))
        nn.init.constant_(self.moving_ii, 1/math.sqrt(2))
        nn.init.constant_(self.moving_ri, 0)
        if self.scale:
            nn.init.constant_(self.gamma_rr, 1/math.sqrt(2))
            nn.init.constant_(self.gamma_ii,1/math.sqrt(2))
        
            nn.init.constant_(self.gamma_ri, 0)
            nn.init.constant_(self.beta, 0)


    def forward(self, input):

        shape_list = list(input.shape)
        ndim = len(shape_list)

        # input data is 2D  batch,num
        if ndim != 4:
            raise ValueError('expected 4D input data, but:')
            print(shape_list)

        input_per = input.permute(0,2,3,1)  # Batch channel height width -> B H W C
        input_dim = int(input_per.shape[-1]/2)
        dim_0 = input_per.shape[0]  # B
        dim_1 = input_per.shape[1]  # H
        dim_2 = input_per.shape[2]  # W
        dim_3 = input_per.shape[3]  # C
        repeat_dim = dim_0*dim_1*dim_2
       

        #input_reshape = input_per.contiguous().view(-1,dim_3)


        input_re = input_per.contiguous().view(-1,dim_3)
        if self.training:
            mu = torch.mean(input_re,dim = 0)
            input_ce = input_re - mu
            var = torch.mean(input_ce ** 2, dim = 0)
            x_norm = input_ce/torch.sqrt(var *2  + self.eps)

            with torch.no_grad():
                self.moving_mean.data = self.momentum * self.moving_mean.data + (1-self.momentum) * mu

                self.moving_rr.data = self.momentum * self.moving_rr.data + (1-self.momentum) * var #Vrr
        else:
            x_norm = (input_re - self.moving_mean)/torch.sqrt(self.moving_rr *2 + self.eps)

        if self.scale:
            real_norm = x_norm[:,:input_dim]
            imag_norm = x_norm[:,input_dim:]

            exg_rr = self.gamma_rr.repeat(repeat_dim,1)
            exg_ii = self.gamma_ii.repeat(repeat_dim,1)
        
            exg_beta = self.beta.repeat(repeat_dim, 1)
        
            real_out = exg_rr * real_norm - exg_ii * imag_norm + exg_beta[:,:input_dim]
            imag_out = exg_ii * real_norm + exg_rr * imag_norm + exg_beta[:,input_dim:]
            out_re = torch.cat((real_out, imag_out), -1)
            out_per = out_re.view(dim_0, dim_1, dim_2, -1)
        else:
            out_per = x_norm.view(dim_0, dim_1, dim_2,-1)
        
        return out_per.permute(0,3,1,2)


