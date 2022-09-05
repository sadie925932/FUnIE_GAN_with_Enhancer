"""
 > Training Type: (seven type)
    1.Ori_Pyramid: Original Pyramid Pooling
    2.Channel: Original Channel Attention Module
    3.Spatial: Original Spatial Attention Module
    4.Channel_Spatial: Original Channel  Attention Module and Spatial Attention Module
    5.Channel_Pyramid: Combine Channel Attention Module and Pyramid Pooling
    6.Spatial_Pyramid: Combine Spatial Attention Module and Pyramid Pooling
    7.Channel_Spatial_Pyramid: Combine Channel Attention Module, Spatial Attention Module and Pyramid Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import *

class Pyramid_pooling(nn.Module) :
    def __init__(self,in_size,in_height,pool_sizes,mode): #in_size -> channel
        super(Pyramid_pooling, self).__init__()
        self.mode = mode # Spatial / Channel / Both / Original
        self.pool_sizes = pool_sizes
        self.in_size = in_size
        self.in_height = in_height

        # For mode='None'and'Channel'
        self.convolution_N = nn.Conv2d(in_size,in_size//4,kernel_size=1) #channel
        self.convolution_last_N = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size*2, 3, kernel_size=3, padding=[1, 1]),
            nn.Tanh()
        )

        # For mode='Spatial'and 'Both'
        self.spatial_pooling_list = nn.ModuleList(
            [SpatailGate(in_size,pool_size) for pool_size in pool_sizes]
        )
        self.convolution = nn.Conv2d(2, 2, kernel_size=1)  # channel
        self.convolution_last = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size+8, 3, kernel_size=3, padding=[1, 1]), # 2*4+in_size`
            nn.Tanh()
        )

    def forward(self,x):
        output = [x]
        if self.mode == 'Original':
            for pool_size in self.pool_sizes:
                out = nn.functional.avg_pool2d(x, pool_size, stride=pool_size, padding=[1, 1], count_include_pad=False)
                out = self.convolution_N(out)
                out = nn.functional.interpolate(out, self.in_height, mode='bilinear', align_corners=False)
                output.append(out)
            output = torch.cat(output, dim=1)
            output = self.convolution_last_N(output)

        elif self.mode == 'Spatial':
            for spatial_pooling in self.spatial_pooling_list:
                out = spatial_pooling(x)
                out = self.convolution(out)
                # out = nn.functional.upsample(out,self.in_height,mode='bilinear') #nn.functional.interpolate
                out = nn.functional.interpolate(out, self.in_height, mode='bilinear', align_corners=False)
                output.append(out)

            output = torch.cat(output, dim=1)
            output = self.convolution_last(output)

        return output

class SpatailGate(nn.Module):
    def __init__(self,in_size,pool_size):
        super(SpatailGate,self).__init__()
        self.avg_pool = nn.AvgPool2d(pool_size,stride=pool_size, padding=[1, 1], count_include_pad=False)
        self.max_pool = nn.MaxPool2d(pool_size,stride=pool_size, padding=[1, 1])
        self.convolution = nn.Conv2d(2,2,kernel_size=5)

    def forward(self,x):
        output = []
        # To Spatial Attention
        m = torch.max(x, 1)[0].unsqueeze(1) #batch*1*256*256
        a = torch.mean(x,1).unsqueeze(1) #batch*1*256*256
        # Pyramid Pooling
        output.append(self.avg_pool(a))
        output.append(self.max_pool(m))
        output = torch.cat(output,dim=1)
        #output = self.convolution(output)

        return output

class Channel_Pyramid(nn.Module):
    def __init__(self,in_size,in_height,pool_sizes):
        super(Channel_Pyramid, self).__init__()

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(in_size,in_size//pool_size,kernel_size=3,stride=1,padding=1)
             for pool_size in pool_sizes]
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=in_height, stride=[in_height, in_height])
        self.max_pool = nn.MaxPool2d(kernel_size=in_height, stride=[in_height, in_height])
        self.conv_list_middle = nn.ModuleList(
            [nn.Conv2d(in_size//pool_size, in_size//pool_size,kernel_size=3,padding=1,stride=1) for pool_size in pool_sizes],
        )
        self.conv_list_last = nn.ModuleList(
            [nn.Conv2d(in_size // pool_size, in_size, kernel_size=3, padding=1, stride=1) for pool_size in pool_sizes],
        )
        self.Relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.upsample = Upsample(320,3)

    #def channel_gate(self,x):

    def forward(self,x):
        output = [x]
        for conv, conv_midd, conv_last in zip(self.conv_list, self.conv_list_middle, self.conv_list_last):
            x_out = conv(x)
            a = self.max_pool(x_out)
            m = self.avg_pool(x_out)
            a = conv_midd(a)
            a = self.Relu(a)
            m = conv_midd(m)
            m = self.Relu(m)
            a = conv_last(a)
            a = self.Sigmoid(a)
            m = conv_last(m)
            m = self.Sigmoid(m)
            out = a + m
            out = out.expand_as(x)
            output.append(out)

        output = torch.cat(output, dim=1)
        output = self.upsample(output)

        return output

class Pyramid(nn.Module):
    def __init__(self,in_size,in_height,pool_sizes):
        super(Pyramid,self).__init__()
        kernel_size = [3,3,5,7]
        self.in_size = in_size
        self.pool_sizes = pool_sizes

        self.conv_1 = nn.ModuleList(
            [nn.Conv2d(in_size,in_size*pool_sizes[i],kernel_size[i],padding=(kernel_size[i] - 1) // 2)
             for i in range(4)]
        )
        self.avg_pool = nn.ModuleList(
            [self.avg_pool_layer(self.in_size * self.pool_sizes[i], self.pool_sizes[i],kernel_size[i])
             for i in range(4)]
        )
        self.max_pool = nn.ModuleList(
            [self.max_pool_layer(self.in_size * self.pool_sizes[i], self.pool_sizes[i],kernel_size[i])
             for i in range(4)]
        )
        self.upsample = Upsample(352,3)

    def avg_pool_layer(self,channel_size,pool_size,kernel_size):
        layers = [nn.AvgPool2d(2, stride=2, count_include_pad=False),
                  nn.Conv2d(channel_size,channel_size//2,kernel_size,padding=(kernel_size-1)//2),
                  nn.ReLU()
                  ]
        return nn.Sequential(*layers)

    def max_pool_layer(self,channel_size,pool_size,kernel_size):
        layers = [nn.MaxPool2d(2, stride=2),
                  nn.Conv2d(channel_size,channel_size//2,kernel_size,padding=(kernel_size-1)//2),
                  nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self,x):
        output = []
        for conv1,avg,max in zip(self.conv_1,self.avg_pool,self.max_pool):
            x_out = conv1(x)
            a_out = avg(x_out)
            m_out = max(x_out)
            out = m_out + a_out
            #out = out.expand_as(x)
            output.append(out)

        output = torch.cat(output, dim=1)
        output = self.upsample(output)

        return output

class Upsample(nn.Module):
    def __init__(self,in_size,out_size):
        super(Upsample,self).__init__()
        self.Upsample = nn.Sequential(
            nn.Upsample(scale_factor=4), # H and W *2
            nn.Conv2d(in_size,out_size,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    def forward(self,x):
        return self.Upsample(x)

class Pyramid_Block(nn.Module):
    def __init__(self,in_size,in_height,pool_sizes,mode):
        super(Pyramid_Block,self).__init__()
        self.mode = mode
        self.in_size = in_size
        self.in_height = in_height
        self.pool_sizes = pool_sizes

        self.Pyramid_pooling = Pyramid_pooling(in_size, in_height, pool_sizes, mode)
        self.Channel_Pyramid = Channel_Pyramid(in_size, in_height, pool_sizes)
        self.Pyramid = Pyramid(in_size, in_height, pool_sizes)

    def forward(self,x):
        if self.mode in ['Spatial','Original']:
            x = self.Pyramid_pooling(x)
        elif self.mode == 'Channel':
            x = self.Channel_Pyramid(x)
        elif self.mode == 'Try2':
            x = self.Pyramid(x)

        return x
