"""
 > Script for testing .pth models  
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output) 
"""
# py libs
import os
import csv
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from Evaluation.measure_uiqm import measure_UIQMs
from Evaluation.meansure_uciqe import measure_UCIQE
from Evaluation.measure_ssim_psnr import SSIMs_PSNRs
import csv

## options
parser = argparse.ArgumentParser()
#parser.add_argument("--inp_dir", type=str, default="/home/cbel/Desktop/Sadie/FUnIE-GAN-master/EUVP/Paired/")
parser.add_argument("--inp_dir", type=str, default="../data/test/")
parser.add_argument("--sample_dir", type=str, default="../data/output/")
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
#parser.add_argument("--model_name", type=str, default="ugan")
#parser.add_argument("--model_path", type=str, default="models/funie_generator.pth")
parser.add_argument("--model_path", type=str, default="models/generator_200.pth")
parser.add_argument("--groundtruth", type=bool, default=True)
opt = parser.parse_args()

dataset = 'UIEB_NYU'
attention = 'False'
pyramid = 'Try2'
architecture = attention+'_'+pyramid
batch_size = 'batch_16'
epoch_num ='200'
model_dir = architecture+'_'+epoch_num
## Remeber to change train model!

csv_path = '/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/quality/FunieGAN/compare.csv'
csv_all_path ='/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/quality/FunieGAN/compare_all.csv'
f_csv = open(csv_path,'a+')
f_all_csv = open(csv_all_path,'a+')
data = ['model','epoch','training','testing','PSNR','SSIM','UCIQE','UIQM']
writer = csv.writer(f_csv)
writer_all = csv.writer(f_all_csv)
csv_detail_path = os.path.join('/home/cbel/Desktop/Sadie/FUnIE-GAN-master/PyTorch_test/checkpoints/FunieGAN/UIEB_NYU/',\
                               'Pyramid_512','test',batch_size,model_dir)


list = ['AL-NG-OVD','CADDY1','CADDY2','DebrisImg1','DebrisImg2','DebrisImg3',
        'DeepSeaImg','Jamaica','HIMB','UIEB_pinyi','underwater_dark','underwater_imagenet','underwater_scenes','test_samples',
        'UIEB','NYU','UIEB_challenge']
#list = ['UIEB_pinyi','underwater_dark','underwater_imagenet','underwater_scenes','test_samples',
#        'UIEB','NYU','UIEB_challenge']

## checks
assert exists(opt.model_path), "model not found"
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch
if opt.model_name.lower()=='funiegan':
    from nets import funiegan
    model = funiegan.GeneratorFunieGAN(attention_mode = attention,pyramid_mode=pyramid)
elif opt.model_name.lower()=='ugan':
    from nets.ugan import UGAN_Nets
    model = UGAN_Nets(base_model='pix2pix').netG
else: 
    # other models
    pass

## model info
#print(torch.load(opt.model_path))
#new_model = torch.load(opt.model_path)
#summary(new_model,input_size=(10,3,256,256))
#print(model)
# summary(model,input_size=(10,3,256,256))
#input()

## load weights
model.load_state_dict(torch.load(opt.model_path),strict=False)
if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 512, 512, 3
transforms_ = [transforms.Resize((img_height, img_width), InterpolationMode.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)
time_list = []
# Scale image for run time
## testing loop
times = []
all_avg_uciqe = [];all_avg_uiqm =[]
all_avg_PSNR = [];all_avg_SSIM =[]
gen_PSNR=0;gen_SSIM=0

for dir in list:
    print("-----%s-----"%(dir))
    # input dataset
    data_dir = os.path.join(str(opt.inp_dir), str(dir))
    if dir in ['underwater_dark','underwater_imagenet','underwater_scenes','test_samples','UIEB','NYU']:
        data_dir = os.path.join(str(opt.inp_dir), str(dir),'test_A')
    gtr_dir = os.path.join(str(opt.inp_dir), str(dir), 'test_B')
    test_files = sorted(glob(join(data_dir, "*.*")))
    # output dir
    output_dir = os.path.join(str(opt.sample_dir),str(dir),opt.model_name,dataset,'512',batch_size,architecture,model_dir)
    os.makedirs(output_dir, exist_ok=True)
    for path in test_files:
        img = Image.open(path)
        (w,h) = img.size
        inp_img = transform(img) #tensor
        inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
        # generate enhanced image
        s = time.time()
        gen_img = model(inp_img)
        resize2original = transforms.Resize([h,w])
        gen_img = resize2original(gen_img)
        times.append(time.time()-s)
        # save output
        #img_sample = torch.cat((inp_img.data, gen_img.data), -1)
        save_image(gen_img, join(output_dir,basename(path)), normalize=True)
        #print ("Tested: %s" % path)

    ## run-time
    if (len(times) > 1):
        print ("\nTotal samples: %d" % len(test_files))
        # accumulate frame processing times (without bootstrap)
        Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
        print("Total time : %f" %(np.sum(times)))
        print("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
        print("Saved generated images in in %s\n" %(output_dir))
    time_list.append(np.sum(times))
    ## Evaluation PSNR and SSIM
    if opt.groundtruth == True :
        gen_SSIM, gen_PSNR = SSIMs_PSNRs(output_dir,gtr_dir)
        print("SSIM : %f"%(np.mean(gen_SSIM)))
        print("PSNR : %f"%(np.mean(gen_PSNR)))
        all_avg_SSIM.append(np.mean(gen_SSIM))
        all_avg_PSNR.append(np.mean(gen_PSNR))

    #output_dir = os.path.join(str(opt.inp_dir),dir,'test_A')

    ## Evaluation UCIQE and UIQM
    gen_uciqe = measure_UCIQE(output_dir, csv_detail_path, im_res=[512, 512],write_csv=True)
    print("UCIQE : %f"%(np.mean(gen_uciqe[0])))
    gen_uiqm = measure_UIQMs(output_dir, csv_detail_path, im_res=[512, 512], write_csv=True)
    print("UIQM  : %f"%(np.mean(gen_uiqm[0])))
    all_avg_uciqe.append(np.mean(gen_uciqe[0]))
    all_avg_uiqm.append(np.mean(gen_uiqm[0]))

    data = [architecture,attention,pyramid,batch_size,epoch_num,dataset,dir,
            np.mean(gen_PSNR),np.mean(gen_SSIM),np.mean(gen_uciqe[0]),np.mean(gen_uiqm[0])]
    writer.writerow(data)


print("-----For All Dataset-----")
if opt.groundtruth == True:
    print("SSIM : Avg = %f , Std = %f" % (np.mean(all_avg_SSIM), np.std(all_avg_SSIM)))
    print("PSNR  : Avg = %f , Std = %f" % (np.mean(all_avg_PSNR), np.std(all_avg_PSNR)))
print("UCIQE : Avg = %f , Std = %f"%(np.mean(all_avg_uciqe),np.std(all_avg_uciqe)))
print("UIQM  : Avg = %f , Std = %f"%(np.mean(all_avg_uiqm),np.std(all_avg_uiqm)))

data_all = [architecture,attention,pyramid,batch_size,epoch_num,dataset,dir,
            np.mean(all_avg_PSNR),np.mean(all_avg_SSIM),np.mean(all_avg_uciqe),np.mean(all_avg_uiqm)]
writer_all.writerow(data)

'''
## UIQMs of the distorted input images
inp_dir = opt.data_dir
inp_uciqe = measure_UCIQE(inp_dir)
inp_uqims = measure_UIQMs(inp_dir)
print ("Input UCIQE >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[0]), np.std(inp_uciqe[0])))
print("\t Chrome >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[1]), np.std(inp_uciqe[1])))
print("\t Luminance >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[2]), np.std(inp_uciqe[2])))
print("\t Saturation >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[3]), np.std(inp_uciqe[3])))
print("--------------------------------------------------------------------------")
print("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims[0]), np.std(inp_uqims[0])))
print("\t UICM >> Mean: {0} std: {1}".format(np.mean(inp_uqims[1]), np.std(inp_uqims[1])))
print("\t UISM >> Mean: {0} std: {1}".format(np.mean(inp_uqims[2]), np.std(inp_uqims[2])))
print("\t UIconM >> Mean: {0} std: {1}".format(np.mean(inp_uqims[3]), np.std(inp_uqims[3])))
print("==========================================================================")
## UCIQEs of the enhanceded output images
gen_dir = opt.sample_dir
gen_uciqe = measure_UCIQE(gen_dir)
gen_uqims = measure_UIQMs(gen_dir)
print ("Enhanced UCIQE >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[0]), np.std(gen_uciqe[0])))
print("\t Chrome >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[1]), np.std(gen_uciqe[1])))
print("\t Luminance >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[2]), np.std(gen_uciqe[2])))
print("\t Saturation >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[3]), np.std(gen_uciqe[3])))
print("--------------------------------------------------------------------------")
print ("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims[0]), np.std(gen_uqims[0])))
print("\t UICM >> Mean: {0} std: {1}".format(np.mean(gen_uqims[1]), np.std(gen_uqims[1])))
print("\t UISM >> Mean: {0} std: {1}".format(np.mean(gen_uqims[2]), np.std(gen_uqims[2])))
print("\t UIconM >> Mean: {0} std: {1}".format(np.mean(gen_uqims[3]), np.std(gen_uqims[3])))
'''