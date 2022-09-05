"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
from .gsop import GSoP
from .pyramid import *
from .Attention import *
from .MPN.MPNCOV import CovpoolLayer
from torchvision.utils import save_image
import torch.nn.functional as F


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True): #size = channel size
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        ## Pyramid Pooling
        #layers.append(pyramid_pooling(128,[4,8,16,32]))
        if bn: layers.append(nn.InstanceNorm2d(out_size, momentum=0.8))
        ## GSoP Layer
        #layers.append(GSoP(planes=int(out_size/4),dim=64).cuda())
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size, momentum=0.8),
            #GSoP Layer
            #GSoP(planes=int(out_size / 4), dim=64).cuda(),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self,attention_mode,pyramid_mode, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        self.attention_mode = attention_mode
        self.pyramid_mode = pyramid_mode
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        #self.att1 = Attention_module(32,256)
        self.down2 = UNetDown(32, 128)
        #self.att1 = Attention_module(128, 128)
        self.down3 = UNetDown(128, 256)
        #self.att1 = Attention_module(256, 64)
        self.down4 = UNetDown(256, 256)
        #self.att1 = Attention_module(256, 32)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            #nn.Conv2d(320, out_channels, 4, padding=1),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        self.Attention_Block = Attention_module(in_size=64,mode=attention_mode)
        self.Pyramid_Block = Pyramid_Block(in_size=64, in_height=256, pool_sizes=[1, 2, 4, 4], mode=pyramid_mode)
        #self.pyramid_block = Pyramid_Block(out_channels)

    def forward(self, x):
        '''
        #if t == 100:
        #    save_image(x[0], "process/gen_layer/x_imgs_process.png", normalize=True)
        g = GSoP(planes=8, dim=64).cuda()
        d1 = self.down1(x) #32x128x128
        y = g(d1)

        #if t == 100:
        #    for i in range(32):
        #        save_image(d1[0,i,:,:], "process/gen_layer/d1_%d_imgs_process.png" % i, normalize=True)
        #print("d2")
        g = GSoP(planes=32, dim=64).cuda()
        d2 = self.down2(y) #128x64x64
        y = g(d2)

        #print("d3")
        g = GSoP(planes=64, dim=64).cuda()
        d3 = self.down3(y) #256x32x32
        y = g(d3)

        #print("d4")
        g = GSoP(planes=64, dim=64).cuda()
        d4 = self.down4(y)
        y = g(d4)

        # print("d5")
        g = GSoP(planes=64, dim=64).cuda()
        d5 = self.down5(y)
        y = g(d5)
        '''
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1) # output size = (batch,64,128,128)
        #output size = (batch,64,256,256)
        if self.attention_mode == 'False' and self.pyramid_mode == 'False':
            u6 = self.final(u45)
        elif self.attention_mode == 'False':
            u6 = self.Pyramid_Block(u45)
        elif self.pyramid_mode == 'False':
            u6 = self.Attention_Block(u45)
            u6 = self.final(u6)
        else:
            u6 = self.Attention_Block(u45)
            u6 = self.Pyramid_Block(u6)

        return u6

class DiscriminatorFunieGAN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            #GSoP
            #layers.append(GSoP(planes=int(out_filters / 4), dim=64).cuda())
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

