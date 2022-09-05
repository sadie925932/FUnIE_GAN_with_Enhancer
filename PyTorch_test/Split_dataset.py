import os
import random
import shutil

def split_dataset(root,sub_dirs):
    for dir in sub_dirs:
        val_A_path = os.path.join(str(root),dir,'rest_A')
        val_B_path = os.path.join(str(root),dir,'rest_B')
        if not os.path.exists(val_B_path) :
            os.mkdir(val_A_path)
            os.mkdir(val_B_path)
        A_path = os.path.join(str(root),dir,'test_A')
        B_path = os.path.join(str(root),dir,'test_B')

        files = os.listdir(A_path)
        #files = random.sample(files,len(files)//50)
        files = random.sample(files,559)
        for file in files:
            shutil.move(f"{A_path}/{file}",val_A_path)
            shutil.move(f"{B_path}/{file}",val_B_path)

if __name__ == '__main__':
    root = "/home/cbel/Desktop/Sadie/IMF/Data"
    sub_dirs = ['NYU']#,'underwater_dark','underwater_scenes']
    split_dataset(root,sub_dirs)




