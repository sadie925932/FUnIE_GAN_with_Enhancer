import torch
from .MPN import MPNCOV
import torch.nn as nn

class GSoP(nn.Module):
    def __init__(self, planes, dim):
        super(GSoP, self).__init__()
        self.ch_dim = dim
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
        self.conv_for_DR = nn.Conv2d(
            planes * 4, self.ch_dim,
            kernel_size=1, stride=1, bias=True
        )
        self.sigmoid = nn.Sigmoid()
        self.row_bn = nn.BatchNorm2d(self.ch_dim)
        self.row_conv_group = nn.Conv2d(
            self.ch_dim, 4 * self.ch_dim,
            kernel_size=(self.ch_dim, 1),
            groups=self.ch_dim, bias=True)
        self.fc_adapt_channels = nn.Conv2d(
            4 * self.ch_dim, planes * 4,
            kernel_size=1, groups=1, bias=True)

    def chan_att(self, x):
        x = self.relu_normal(x)
        x = self.conv_for_DR(x)
        x = self.bn_for_DR(x)
        x = self.relu(x)

        x = MPNCOV.CovpoolLayer(x)  # Nxdxd
        x = x.view(x.size(0), x.size(1), x.size(2), 1).contiguous()  # Nxdxdx1

        x = self.row_bn(x)
        x = self.row_conv_group(x)  # Nx512x1x1
        #print(x.size())

        x = self.fc_adapt_channels(x)  # NxCx1x1
        x = self.sigmoid(x)  # NxCx1x1
        #print(x.size())

        return x

    def forward(self, x):
        pre_att = x
        att = self.chan_att(x)
        out = pre_att * att

        return out
'''
class GSoP_Net2(nn.Module):
    def __init__(self,GSoP_mode):
        if GSoP_mode == 1:
            self.avgpool = nn.AvgPool2d(14, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            print("GSoP-Net1 generating...")
        else:
            self.isqrt_dim = 256
            self.layer_reduce = nn.Conv2d(512 * block.expansion, self.isqrt_dim, kernel_size=1, stride=1, padding=0,
                                          bias=False)
            self.layer_reduce_bn = nn.BatchNorm2d(self.isqrt_dim)
            self.layer_reduce_relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(int(self.isqrt_dim * (self.isqrt_dim + 1) / 2), num_classes)
            print("GSoP-Net2 generating...")

    def forward(self,x):
        if self.GSoP_mode == 1:
            x = self.avgpool(x)
        else:
            x = self.layer_reduce(x)
            x = self.layer_reduce_bn(x)
            x = self.layer_reduce_relu(x)

            x = MPNCOV.CovpoolLayer(x)
            x = MPNCOV.SqrtmLayer(x, 3)
            x = MPNCOV.TriuvecLayer(x)
'''
