"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from csv import writer
from ntpath import basename
## local libs
import pandas as pd
import os
from .uqim_utils import getUIQM


def measure_UIQMs(dir_name,csv_path, im_res=(250, 250),write_csv=False):
    paths = sorted(glob(join(dir_name, "*.*")))
    data = pd.DataFrame(columns=['image', 'uiqm', 'uicm', 'uism', 'uiconm'])
    uiqms = []
    uicms = []
    uisms = []
    uiconms = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm,uicm,uism,uiconm = getUIQM(np.array(im))
        head_tail = os.path.split(str(img_path))
        data = data.append({"image": head_tail[1],
                            "uiqm": uiqm,
                            "uicm": uicm,
                            "uism": uism,
                            "uiconm": uiconm}, ignore_index=True)
        uiqms.append(uiqm)
        uicms.append(uicm)
        uisms.append(uism)
        uiconms.append(uiconm)

    # save average
    data = data.append({"image": 'avg',
                        "uiqm" : np.mean(np.array(uiqms)),
                        "uicm": np.mean(np.array(uisms)),
                        "uism" : np.mean(np.array(uisms)),
                        "uiconm" : np.mean(np.array(uiconms))},ignore_index=True)
    if write_csv == True:
        #csv_path = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/quality/FunieGAN/"
        csv_name = "UIQM_all_result_"+str(im_res[0])+".xlsx"
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

    return np.array(uiqms), np.array(uicms), np.array(uisms), np.array(uiconms)

def com(pth_a,pth_b,dir_list):

    sum_more=[]
    sum_equal=[]
    sum_less=[]

    for dir in dir_list :
        df_a = pd.read_excel(pth_a,sheet_name = dir,usecols=['uiqm'])
        df_b = pd.read_excel(pth_b,sheet_name = dir,usecols=['uiqm'])

        rows,cols = df_a.shape
        x = 0;y = 0;z = 0
        for i in range(rows) :
            if df_a['uiqm'][i] > df_b['uiqm'][i] :
                x = x + 1
            elif df_a['uiqm'][i] == df_b['uiqm'][i] :
                y = + 1
            else :
                z = z + 1
        sum_more.append(x)
        sum_equal.append(y)
        sum_less.append(z)

    print(sum_more,sum_equal,sum_less)
    print(sum(sum_more),sum(sum_equal),sum(sum_less))

"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
'''
#inp_dir = "/home/cbel/Desktop/Sadie/test_data/GT"
#inp_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/test/AL-NG-OVD/"
inp_dir = "/home/cbel/Desktop/Sadie/dehazing/materials/UIEB"
inp_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/underwater_dark/validation_A"

## UIQMs of the distorted input images
inp_uiqms = measure_UIQMs(inp_dir)
print("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uiqms[0]), np.std(inp_uiqms[0])))
print("\t UICM >> Mean: {0} std: {1}".format(np.mean(inp_uiqms[1]), np.std(inp_uiqms[1])))
print("\t UISM >> Mean: {0} std: {1}".format(np.mean(inp_uiqms[2]), np.std(inp_uiqms[2])))
print("\t UIconM >> Mean: {0} std: {1}".format(np.mean(inp_uiqms[3]), np.std(inp_uiqms[3])))
input()

## UIQMs of the enhanceded output images

#gen_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/output/AL-NG-OVD"
#gen_dir = "/home/cbel/Desktop/Sadie/pytorch-CycleGAN-and-pix2pix/results/euvp_org_net_cyclegan/test_latest/recB"
gen_dir = "/home/cbel/Desktop/Sadie/MLFcGAN/infer_result/UIEB"
gen_uiqms = measure_UIQMs(gen_dir)
print ("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uiqms[0]), np.std(gen_uiqms[0])))
print("\t UICM >> Mean: {0} std: {1}".format(np.mean(gen_uiqms[1]), np.std(gen_uiqms[1])))
print("\t UISM >> Mean: {0} std: {1}".format(np.mean(gen_uiqms[2]), np.std(gen_uiqms[2])))
print("\t UIconM >> Mean: {0} std: {1}".format(np.mean(gen_uiqms[3]), np.std(gen_uiqms[3])))
'''
'''
all_avg = []
gen_dir = "/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/output/"
list = ['AL-NG-OVD','CADDY1','CADDY2','DebrisImg1','DebrisImg2','DebrisImg3',
        'DeepSeaImg','Jamaica','HIMB','UIEB']
#list = ["UIEB"]
for dir in list :
    img_path = os.path.join(str(gen_dir),str(dir),'funie','dis_GSoP_pyramid','60')
    gen_uiqm = measure_UIQMs(img_path)
    all_avg.append(np.mean(gen_uiqm[0]))

print(np.mean(all_avg))
print(np.std(all_avg))
print(all_avg)

## For comparision
dir_list = ['UGAN','MLFcGAN']
qua_dir = '/home/cbel/Desktop/Sadie/FUnIE-GAN-master/data/quality/'
for dir in dir_list :
    pth_a = os.path.join(str(qua_dir),str(dir),'UIQM_all_result.xlsx')
    pth_b = os.path.join(str(qua_dir),str(dir),'UIQM_all_result_500.xlsx')
    print(dir)
    com(pth_a,pth_b,list)
'''

