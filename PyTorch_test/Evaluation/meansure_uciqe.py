import os.path

import cv2
import numpy as np
from skimage import io, color, filters
from PIL import Image, ImageOps
from glob import glob
from os.path import join
import csv
import pandas as pd
import openpyxl
import math as m

def nmetrics(x):
    lab = color.rgb2lab(x)

    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]

    #1st term
    chroma = np.sqrt((a**2 + b**2))
    uc = np.mean(chroma)
    sc = np.sqrt(np.mean(np.mean(chroma**2 - uc**2)))

    #2nd term
    conl = np.max(l) - np.min(l)
    #top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    #sl = np.sort(l,axis=None)
    #isl = sl[::-1]
    #conl = np.mean(isl[::top])-np.mean(sl[::top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    return uciqe

def nmetrics2(x):
    # img_bgr = cv2.imread(x)        # Used to read image files
    img_bgr = x
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    # print(np.mean(img_lum))
    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)
    # print(np.cumsum(hist),np.sum(hist))
    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    # print(np.mean(cdf))
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)

    return quality_val,var_chr,con_lum,aver_sat

def measure_UCIQE(dir_name, csv_path, im_res=(250,250),write_csv=False): #org : 250,250

    paths = sorted(glob(join(dir_name, "*.*")))
    data = pd.DataFrame(columns=['image','uciqe','chr','lum','sat'])
    uciqes = []
    chrs = []
    lums = []
    sats = []

    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uciqe,chr,lum,sat = nmetrics2(np.array(im))
        head_tail = os.path.split(str(img_path))
        data = data.append({"image": head_tail[1],
                    "uciqe" : uciqe,
                     "chr": chr,
                     "lum" : lum,
                     "sat" : sat},ignore_index=True)
        #print(data)
        uciqes.append(uciqe)
        chrs.append(chr)
        lums.append(lum)
        sats.append(sat)
    # save average
    data = data.append({"image": 'avg',
                        "uciqe" : np.mean(np.array(uciqes)),
                        "chr": np.mean(np.array(chrs)),
                        "lum" : np.mean(np.array(lums)),
                        "sat" : np.mean(np.array(sats))},ignore_index=True)
    if write_csv == True:
        #csv_path = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/quality/FunieGAN/"
        csv_name = "UCIQE_all_result_"+str(im_res[0])+".xlsx"
        csv_path = os.path.join(str(csv_path),str(csv_name))
        head_tail = dir_name.split("/")
        if head_tail[3] == 'AL-NG-OVD' :
            writer = pd.ExcelWriter(csv_path,
                                    if_sheet_exits = 'new',engine='openpyxl', mode='w')
        else :
            writer = pd.ExcelWriter(csv_path,
                                    if_sheet_exits = 'new',engine='openpyxl',mode = 'a')
        data.to_excel(writer,sheet_name=str(head_tail[3]),index = False)
        writer.save()

    return np.array(uciqes), np.array(chrs), np.array(lums), np.array(sats)

def com(pth_a,pth_b,dir_list):

    sum_more=[]
    sum_equal=[]
    sum_less=[]

    for dir in dir_list :
        df_a = pd.read_excel(pth_a,sheet_name = dir,usecols=['uciqe'])
        df_b = pd.read_excel(pth_b,sheet_name = dir,usecols=['uciqe'])

        rows,cols = df_a.shape
        x = 0;y = 0;z = 0
        for i in range(rows) :
            if df_a['uciqe'][i] > df_b['uciqe'][i] :
                x = x + 1
            elif df_a['uciqe'][i] == df_b['uciqe'][i] :
                y = + 1
            else :
                z = z + 1
        sum_more.append(x)
        sum_equal.append(y)
        sum_less.append(z)

    print(sum_more,sum_equal,sum_less)
    print(sum(sum_more),sum(sum_equal),sum(sum_less))



#inp_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/test/AL-NG-OVD/"
#inp_dir = "/home/cbel/Desktop/Sadie/EPDN-master/datasets/Mix/train_A"
inp_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/EUVP/Paired/underwater_imagenet/validation_A"
'''
## UIQMs of the distorted input images
inp_uciqe = measure_UCIQE(inp_dir)
print ("Input UCIQE >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[0]), np.std(inp_uciqe[0])))
print("\t Chrome >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[1]), np.std(inp_uciqe[1])))
print("\t Luminance >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[2]), np.std(inp_uciqe[2])))
print("\t Saturation >> Mean: {0} std: {1}".format(np.mean(inp_uciqe[3]), np.std(inp_uciqe[3])))

## UIQMs of the enhanceded output images

#gen_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/output/AL_NG_OVD/full_GSoP/100"
#gen_dir = "/home/cbel/Desktop/Sadie/MLFcGAN/infer_result/Mix"
gen_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/output/UIEB/ugan/60"

gen_uciqe = measure_UCIQE(gen_dir)
print ("Enhanced UCIQE >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[0]), np.std(gen_uciqe[0])))
print("\t Chrome >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[1]), np.std(gen_uciqe[1])))
print("\t Luminance >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[2]), np.std(gen_uciqe[2])))
print("\t Saturation >> Mean: {0} std: {1}".format(np.mean(gen_uciqe[3]), np.std(gen_uciqe[3])))
'''
'''
gen_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/output/"
gen_dir = '/home/cbel/Desktop/Sadie/FUnIE-GAN-master/EUVP/Paired/'
#list = ['AL-NG-OVD','CADDY1','CADDY2','DebrisImg1','DebrisImg2','DebrisImg3',
#        'DeepSeaImg','Jamaica','HIMB','UIEB']
list = ['underwater_dark','underwater_imagenet','underwater_scenes']
all_avg = []

def test_dataset_UCIQE(img_path,eval_size):
    #for dir in list :
        #img_path = os.path.join(str(gen_dir),str(dir),'funie','dis_GSoP_pyramid','60')
    gen_uciqe = measure_UCIQE(img_path,im_res=eval_size)
    all_avg.append(np.mean(gen_uciqe[0]))
    print("UCIQE")
    print(np.mean(all_avg))
    print(np.std(all_avg))
    print(all_avg)

## For comparision
dir_list = ['UGAN','MLFcGAN']
qua_dir = '/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/quality/'
for dir in dir_list :
    pth_a = os.path.join(str(qua_dir),str(dir),'UCIQE_all_result.xlsx')
    pth_b = os.path.join(str(qua_dir),str(dir),'UCIQE_all_result_500.xlsx')
    print(dir)
    com(pth_a,pth_b,list)

'''