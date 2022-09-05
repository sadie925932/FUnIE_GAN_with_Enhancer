import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.conv = nn.Conv2d(2,1,kernel_size=kernel_size,stride=1,padding=[3,3],dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(1, momentum=0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x) #2*128*128
        x_out = self.conv(x_compress) #1*128*128
        x_out = self.bn(x_out)
        x_out = self.relu(x_out)
        scale = self.sigmoid(x_out) # broadcasting
        return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Channel_Attention(nn.Module):
    def __init__(self,in_size,in_height):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=in_height, stride=[in_height,in_height])
        self.max_pool = nn.MaxPool2d(kernel_size=in_height, stride=[in_height,in_height])
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_size,in_size//4),
            nn.ReLU(),
            nn.Linear(in_size//4,in_size)
        )

    def forward(self,x):
        a = self.avg_pool(x)
        a = self.mlp(a)
        m = self.max_pool(x)
        m = self.mlp(m)

        output = a + m
        scale = torch.sigmoid(output).unsqueeze(2).unsqueeze(3).expand_as(x)
        # scale  = (batch,64,128,128)

        return x * scale

class Attention_module(nn.Module):
    def __init__(self,in_size,in_height=128,mode='Both'):
        super(Attention_module,self).__init__()
        self.mode = mode
        self.Channel = Channel_Attention(in_size, in_height)
        self.Spatial = Spatial_Attention()

    def forward(self,x):
        if self.mode == 'Channel':
            x_out = self.Channel(x)
        if self.mode == 'Spatial':
            x_out = self.Spatial(x)
        elif self.mode == 'Both':
            x_out = self.Channel(x)
            x_out = self.Spatial(x_out)

        return x_out