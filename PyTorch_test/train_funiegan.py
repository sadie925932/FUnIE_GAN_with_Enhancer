"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
# py libs
import os
import sys
import matplotlib.pyplot as plt
import yaml
import time
import datetime
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from alive_progress import alive_bar
from torchvision import datasets
from torchinfo import summary
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from Evaluation.meansure_uciqe import *
from Evaluation.imqual_utils import getSSIM, getPSNR
from nets.gsop import GSoP
from nets.MPN.MPNCOV import Covpool, CovpoolLayer
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage , GetNewValImage

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_uieb.yaml")
#parser.add_argument("--cfg_file", type=str, default="configs/train_ufo.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
parser.add_argument("--attention_mode", type=str, default='False') # Channel/Spatial/Both/False
parser.add_argument("--pyramid_mode",type=str,default='Try2') #Original/Spatial/Channel/False
args = parser.parse_args()

print("Attention Mode:%s"%(args.attention_mode))
print("Pyramid Mode:%s"%(args.pyramid_mode))

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2 
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"] 
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"] #256
img_height = cfg["im_height"] #256
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

"""------------File-----------------"""
## For Testing use : checkpoints_dir
dir = args.attention_mode + '_' +args.pyramid_mode + '_' + str(num_epochs)

## create dir for model and validation data
samples_dir = os.path.join("samples/FunieGAN/", dataset_name)
checkpoint_dir = os.path.join(("/home/cbel/Desktop/Sadie/FUnIE-GAN-master/PyTorch_test/checkpoints/FunieGAN/"
                               "%s/Pyramid_512/test/batch_%d"%(dataset_name,batch_size)),dir)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

path = os.path.join(checkpoint_dir,"output.txt")
path_all = os.path.join(checkpoint_dir,"loss.txt")
path_val = os.path.join(checkpoint_dir,"val_loss.txt")
f = open(path,"a+")
#for plotting
f_all = open(path_all,'a+')
f_val = open(path_val,'a+')

best_val_uciqe = 0
best_epoch = 0
best_val_loss = 1000
xlabel=[];ylabel=[];y2label=[]

""" FunieGAN specifics: loss functions and patch-size
-----------------------------------------------------"""
Adv_cGAN = torch.nn.MSELoss()
L1_G  = torch.nn.L1Loss() # similarity loss (l1)
L_vgg = VGG19_PercepLoss() # content loss (vgg)
lambda_1, lambda_con = 7, 3 # 7:3 (as in paper)
patch = (1, img_height//16, img_width//16) # 16x16 for 256x256

# Initialize generator and discriminator
generator = GeneratorFunieGAN(attention_mode=args.attention_mode,pyramid_mode=args.pyramid_mode)
discriminator = DiscriminatorFunieGAN()

# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    Adv_cGAN.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    #generator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, args.epoch)))
    #discriminator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch)))
    generator.load_state_dict(torch.load(os.path.join(checkpoint_dir,("generator_%d.pth" % args.epoch))))
    discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir,("discriminator_%d.pth" % args.epoch))))
    # read validation loss
    f_val = open(path_val, 'r+')
    i = 0;
    for line in f_val:
        xlabel.append(int(i))
        i = i+1
        ylabel.append(float(line[:-1]))
    print ("Loaded model from epoch %d" %(epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)

new_val_dataloader = DataLoader(
    GetNewValImage(dataset_path, dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

plt.ion() #开启interactive模式
## Training pipeline
for epoch in range(epoch, num_epochs):
    # Epoch Information
    print("[ Epoch",epoch,"]:",datetime.datetime.now())
    #f.write("\n")
    Dloss = 0;Gloss = 0;Advloss = 0
    x=0;y=0;z=0
    with alive_bar(len(dataloader), force_tty=True, spinner='notes', bar='filling') as bar:
        for i, batch in enumerate(dataloader):
            # Model inputs
            imgs_distorted = Variable(batch["A"].type(Tensor))
            imgs_good_gt = Variable(batch["B"].type(Tensor))
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

            ## Train Discriminator
            optimizer_D.zero_grad()
            imgs_fake = generator(imgs_distorted) #size:8*3*256*256
            pred_real = discriminator(imgs_good_gt, imgs_distorted) # size:8*1*16*16
            loss_real = Adv_cGAN(pred_real, valid) #MSE_Loss
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_fake = Adv_cGAN(pred_fake, fake)
            # Total loss: real + fake (standard PatchGAN)
            loss_D = 0.5 * (loss_real + loss_fake) * 10.0 # 10x scaled for stability
            loss_D.backward()
            optimizer_D.step()
            x = x + loss_D.item()

            ## Train Generator
            optimizer_G.zero_grad()
            imgs_fake = generator(imgs_distorted)
            pred_fake = discriminator(imgs_fake, imgs_distorted)
            loss_GAN = Adv_cGAN(pred_fake, valid) # GAN loss
            loss_1 = L1_G(imgs_fake, imgs_good_gt) # similarity loss
            loss_con = L_vgg(imgs_fake, imgs_good_gt)# content loss
            # Total loss (Section 3.2.1 in the paper)
            loss_G = loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con
            loss_G.backward()
            optimizer_G.step()
            y = y + loss_G.item()
            z = z + loss_GAN.item()

            ## Print log
            if not i%50:
                sys.stdout.write("[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                                  %(
                                    epoch, num_epochs, i, len(dataloader),
                                    loss_D.item(), loss_G.item(), loss_GAN.item(),
                                   )
                )
                f.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                                 % (
                                     epoch, num_epochs, i, len(dataloader),
                                     loss_D.item(), loss_G.item(), loss_GAN.item(),
                                 )
                )
            ## If at sample interval save image
            batches_done = epoch * len(dataloader) + i
            '''
            if batches_done % val_interval == 0: #val_interval = 1000
                imgs = next(iter(val_dataloader))
                imgs_val = Variable(imgs["val"].type(Tensor))
                imgs_gen = generator(imgs_val)
                img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
                save_image(img_sample, "samples/FunieGAN/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)
            '''
            # Bar update
            bar()
        #For Plotting
        Dloss = x/len(dataloader)
        Gloss = y/len(dataloader)
        Advloss = z/len(dataloader)
        f_all.write("%f,%f,%f\n"%(Dloss,Gloss,Advloss))
        f_all.flush()

    ## Best Validation Model
    # split tensor
    if epoch not in xlabel:
        avg_val_uciqe = 0
        val_loss = 0
        #im = next(iter(new_val_dataloader))
        for j,im in enumerate(new_val_dataloader):
            im_val = Variable(im["val_A"].type(Tensor))
            im_val_gt = Variable(im["val_B"].type(Tensor))
            im_val_gen = generator(im_val)
            val_adv_loss = Adv_cGAN(im_val_gen,im_val_gt)
            val_L1_loss = L1_G(im_val_gen,im_val_gt)
            val_vgg_loss = L_vgg(im_val_gen,im_val_gt)
            val_loss = val_adv_loss+ lambda_1 * val_L1_loss + lambda_con * val_vgg_loss
            val_loss = val_loss.item()
            y2label.append(val_loss)

        val_loss = np.mean(y2label)
        val_loss = val_loss/(j+1)
        print("Validation Average loss : %f" % (val_loss))
        plt.xlabel("epoch")
        plt.ylabel("val loss")
        xlabel.append(epoch)
        #val_loss = val_loss.numpy()
        f_val.write("%f\n"%val_loss)
        f_val.flush()
        ylabel.append(val_loss)
        plt.plot(xlabel,ylabel,label='Val loss')
        #plt.show()
        plt.pause(5)

        if val_loss < best_val_loss :
            best_val_loss = val_loss
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir,"best_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir,"best_discriminator.pth"))
            best_epoch = epoch
        sys.stdout.write("Best Generator in %d\n" % (best_epoch))

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir,("generator_%d.pth" % epoch)))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir,("discriminator_%d.pth" % epoch)))

plt.ioff()
plt.savefig(os.path.join(checkpoint_dir,'val_loss_%d'%(best_epoch)))
#plt.show()
f_all.close()
f.close()
f_val.close()
print("Attention Mode:%s"%(args.attention_mode))
print("Pyramid Mode:%s"%(args.pyramid_mode))
