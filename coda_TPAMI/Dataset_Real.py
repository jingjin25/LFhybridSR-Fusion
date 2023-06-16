import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import h5py
import numpy as np
import random
import cv2
from scipy import misc
from math import ceil
from PIL import Image


class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()

        hf = h5py.File(opt.dataset_path)
        self.img_label = hf.get('img_label')  # [N,ah,aw,3,h,w]
        self.img_LR = hf.get('img_LR_{}'.format(opt.scale))  # [N,ah,aw,3,h/8,w/8]

        self.scale = opt.scale
        self.psize = opt.patch_size
        self.an = opt.angular_num

    def __getitem__(self, index):

        # get one item
        label = self.img_label[index]  # [ah,aw,3,h,w]
        lr = self.img_LR[index]  # [ah,aw,3,h/8,w/8]

        # crop to patch
        H = label.shape[3]
        W = label.shape[4]

        # print('H: ', H)
        # print('W: ', W)
        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)
        label = label[:, :, :, x:x + self.psize, y:y + self.psize]  # [ah,aw,3,ph,pw]
        lr = lr[:, :, :, x//self.scale:x//self.scale + self.psize//self.scale, y//self.scale:y//self.scale + self.psize//self.scale]  # [ah,aw,3,ph/8,pw/8]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            label = np.flip(np.flip(label, 0), 3)
            lr = np.flip(np.flip(lr, 0), 3)
        if np.random.rand(1) > 0.5:
            label = np.flip(np.flip(label, 1), 4)
            lr = np.flip(np.flip(lr, 1), 4)


        # rotate
        r_ang = np.random.randint(1, 5)
        label = np.rot90(label, r_ang, (3, 4))
        label = np.rot90(label, r_ang, (0, 1))
        lr = np.rot90(lr, r_ang, (3, 4))
        lr = np.rot90(lr, r_ang, (0, 1))


        trans_toTensor = transforms.ToTensor()

        # label: RGB to ycbcr
        label_y = torch.zeros(self.an, self.an, self.psize, self.psize)

        for i in range(self.an):
            for j in range(self.an):
                sub = label[i, j].transpose(1, 2, 0)
                sub = Image.fromarray(sub.astype('uint8')).convert('RGB')
                sub_ycbcr = sub.convert('YCbCr')
                label_y[i, j] = trans_toTensor(sub_ycbcr)[0]

        ind = ceil(self.an / 2.) - 1
        hr = label_y[ind, ind, :, :].unsqueeze(0)

        label_y = label_y.view(self.an*self.an, self.psize, self.psize)

        # LR color augmentation
        lr_ct_y = torch.zeros(self.an, self.an, self.psize//self.scale, self.psize//self.scale)

        for i in range(self.an):
            for j in range(self.an):
                sub = lr[i, j].transpose(1, 2, 0)
                sub = Image.fromarray(sub.astype('uint8')).convert('RGB')
                # sub.save('img_label_{}{}.png'.format(i, j))
                trans_color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                sub_tc = trans_color(sub)
                # sub_tc.save('img_label_trans_{}{}.png'.format(i, j))
                sub_tc_ycbcr = sub_tc.convert('YCbCr')
                lr_ct_y[i, j] = trans_toTensor(sub_tc_ycbcr)[0]
        lr_ct_y = lr_ct_y.view(self.an*self.an, self.psize//8, self.psize//8)

        return label_y, lr_ct_y, hr

    def __len__(self):
        return self.img_label.shape[0]


