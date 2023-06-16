import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import h5py
import numpy as np
import random
# import cv2
# from scipy import misc
from math import ceil
# from PIL import Image


class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()

        hf = h5py.File(opt.dataset_path)
        self.img_label = hf.get('img_label')  # [N,ah,aw,h,w]
        self.img_LR = hf.get('img_LR_{}'.format(opt.scale))

        self.scale = opt.scale
        self.psize = opt.patch_size
        self.an = opt.angular_num

    def __getitem__(self, index):

        # get one item
        label = self.img_label[index]  # [ah,aw,3,h,w]
        lr = self.img_LR[index]  # [ah,aw,h/s,w/s]

        # crop to patch
        H = label.shape[2]
        W = label.shape[3]

        # print('H: ', H)
        # print('W: ', W)
        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        label = label[:self.an, :self.an,  x:x + self.psize, y:y + self.psize]  # [ah,aw,3,ph,pw]
        lr = lr[:self.an, :self.an,  x // self.scale:x // self.scale + self.psize // self.scale, y // self.scale:y //self.scale + self.psize // self.scale]  # [ah,aw,3,ph/8,pw/8]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            label = np.flip(np.flip(label, 0), 2)
            lr = np.flip(np.flip(lr, 0), 2)
        if np.random.rand(1) > 0.5:
            label = np.flip(np.flip(label, 1), 3)
            lr = np.flip(np.flip(lr, 1), 3)

        # rotate
        r_ang = np.random.randint(1, 5)
        label = np.rot90(label, r_ang, (2, 3))
        label = np.rot90(label, r_ang, (0, 1))
        lr = np.rot90(lr, r_ang, (2, 3))
        lr = np.rot90(lr, r_ang, (0, 1))

        #
        ind = ceil(self.an/2.)-1
        hr = label[ind, ind, :, :]

        # to tensor
        label = label.reshape(-1, self.psize, self.psize)     # [an,ph,pw]
        hr = hr.reshape(-1, self.psize, self.psize)
        lr = lr.reshape(-1, self.psize//self.scale, self.psize//self.scale) # [an,ph,pw]

        label = torch.from_numpy(label.astype(np.float32)/255.0)
        lr = torch.from_numpy(lr.astype(np.float32)/255.0)
        hr = torch.from_numpy(hr.astype(np.float32)/255.0)

        return label, lr, hr

    def __len__(self):
        return self.img_label.shape[0]


