import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

def animate(i) :
    plt.clf()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    path = ('./checkpoints/FunieGAN/UIEB_NYU/Pyramid_512/batch_16/Spatial_Try_201/loss.txt')
    f = open(path,'r')
    lines = f.readlines()
    Dloss,Gloss,Aloss = [],[],[]
    for line in lines:
        Dloss.append(float(line.split(',')[0]))
        Gloss.append(float(line.split(',')[1]))
        Aloss.append(float(line.split(',')[2]))
    x = np.linspace(1,len(Dloss),len(Dloss))
    plt.xticks(x)
    plt.plot(x,Gloss,label='G Loss')
    plt.plot(x,Dloss,label='D Loss')
    #plt.plot(x,Aloss,label='Adv Loss')
    plt.legend(loc='upper right')
    plt.xlim(1,len(x))
    axes = plt.gca()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig('./checkpoints/FunieGAN/UIEB_NYU/Pyramid_512/batch_16/Spatial_Try_201/loss_fig')

def animate_val(i):
    plt.clf()
    plt.xlabel("epoch")
    plt.ylabel("val_loss")
    path = ('val_loss.txt')
    f = open(path,'r')
    lines = f.readlines()
    Gloss = []
    for line in lines:
        Gloss.append(float(line.split(',')[0]))
    x = np.linspace(1,len(Gloss),len(Gloss))
    plt.xticks(x)
    plt.plot(x,Gloss,label='val Loss')

    plt.legend(loc='upper left')
    plt.xlim(1,len(x))
    axes = plt.gca()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig('val_loss_200')

def animation_plot_loss():
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xlim(0,10)
    #plt.figure(figsize=(20, 16))
    ani = FuncAnimation(plt.gcf(),animate,interval=1000)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    animation_plot_loss()