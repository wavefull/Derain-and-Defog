import os
import cv2
import torch
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import settings 
from utils import  edge_compute

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num*100

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
       
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        
        if settings.aug_data:
            O, L = self.crop(img_pair, aug=True)
            O, L = self.flip(O, L)
            O, L = self.rotate(O, L)
        else:
            O, L = self.crop(img_pair, aug=False)
       
        O = np.transpose(O, (2, 0, 1))
        L = np.transpose(B, (2, 0, 1))   
        sample = {'O': O,'L': L,'idx': idx}
        return sample

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
             mini = - 1 / 4 * self.patch_size
             maxi =   1 / 4 * self.patch_size + 1
             p_h = patch_size + self.rand_state.randint(mini, maxi)
             p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
             p_h, p_w = patch_size, patch_size
           
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        L = img_pair[r: r+p_h, c+w: c+p_w+w]
        B = img_pair[r: r+p_h, c: c+p_w]
        if aug:
            L = cv2.resize(L, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))
       
        s=L[:,:,0]
        s=s[:,:,np.newaxis]
        s =  np.tile(s, [1, 1, 3])
        a=L[:,:,1]
        a=a[:,:,np.newaxis]
        a = np.tile(a, [1, 1, 3])
        t=L[:,:,2]
        t=t[:,:,np.newaxis]
        t = np.tile(t, [1, 1, 3])
        rainimage=B*t+(s+a)*(1-t)
        O = rainimage
        return O, L

    def flip(self, O, L):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            L = np.flip(B, axis=1)
        return O, L

    def rotate(self, O, L):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        L = cv2.warpAffine(L, M, (patch_size, patch_size))
        return O, L


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
      
        h, ww, c = img_pair.shape
        w = ww
        O = np.transpose(img_pair[0:h, 0:w], (2, 0, 1))
        sample = {'O': O,'idx': idx}

        return sample
    def get_name(self, idx):
        return self.img_files[idx % self.file_num].split('.')[0]


if __name__ == '__main__':
    dt = TrainValDataset('val')
    print('TrainValDataset')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())

    print()
    print('ShowDataset')
    dt = ShowDataset('test')
    for i in range(10):
        smp = dt[i]
        for k, v in smp.items():
            print(k, v.shape, v.dtype, v.mean())
