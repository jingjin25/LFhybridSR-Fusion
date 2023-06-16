# test on a certain epoch to record details
# records PSNR/SSIM of SRfusion SRcopy SRwawrp for each LFI 

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import argparse
import numpy as np
import os
from os import listdir
from os.path import join
from math import ceil
from scipy import io

import math
import copy
import pandas as pd
 

import h5py
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from scipy import misc
from skimage.measure import compare_ssim  

from Model_Fusion import FusionNet


#--------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------------------------------------------------------------------------#
# Test settings
parser = argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR -- test only")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--model_dir", type=str, default="", help="model dir")

parser.add_argument("--scale", type=int, default=2, help="SR factor")
parser.add_argument("--train_dataset", type=str, default="",help="dataset for training ")
parser.add_argument("--test_dataset", type=str, default="",help="dataset for test")
parser.add_argument("--test_path", type=str, default="",help="dataset for test")
parser.add_argument("--angular_num", type=int, default=9, help="Size of angular dim")
parser.add_argument("--smooth", type=float, default=0.01, help="smooth loss weight")

parser.add_argument("--epoch", type=int, default=1, help="start epoch for test")

parser.add_argument("--save_img", type=int, default=0,help="save image or not")
# parser.add_argument("--lfi_info", type=int, default=1,help="save results for each testing LF image")

parser.add_argument("--crop", type=int, default=0,help="crop the image into patches when not enough memory")
opt = parser.parse_args()
print(opt)
#-----------------------------------------------------------------------------------#   
class DatasetFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(opt.test_path)

        self.LR = hf.get('/LR') #[N,ah,aw,3,h/s,w/s]
        self.HR = hf.get('/HR') #[N,1,1,3,h,w]

        self.scale = opt.scale
        self.an = opt.angular_num

    def __getitem__(self, index):

        h = self.HR.shape[4]
        w = self.HR.shape[5]


        trans_toTensor = transforms.ToTensor()

        lr = self.LR[index]
        lr_ycbcr = torch.zeros(an, an, 3, h//self.scale, w//self.scale)
        for i in range(an):
            for j in range(an):
                sub = lr[i, j].transpose(1, 2, 0)
                sub = Image.fromarray(sub.astype('uint8')).convert('RGB')
                sub_ycbcr = sub.convert('YCbCr')
                lr_ycbcr[i, j] = trans_toTensor(sub_ycbcr)
        lr_ycbcr = lr_ycbcr.view(an*an, 3, h//self.scale, w//self.scale)


        hr = self.HR[index]
        hr_ycbcr = Image.fromarray(hr.reshape(3, h, w).transpose(1, 2, 0).astype('uint8')).convert('RGB').convert('YCbCr')
        hr_ycbcr = trans_toTensor(hr_ycbcr)


        return lr_ycbcr, hr_ycbcr

    def __len__(self):
        return self.HR.shape[0]
#-----------------------------------------------------------------------------------#        

#-------------------------------------------------------------------------------#
if opt.model_dir == "":
    model_dir = 'model_{}_x{}_sm{}_lr{}_step{}x{}'.format(opt.train_dataset,opt.scale,opt.smooth,opt.lr,opt.step,opt.reduce)
else:
    model_dir = opt.model_dir

if not os.path.exists('quan_results'):
    os.makedirs('quan_results')

if not os.path.exists(model_dir):
    print('model folder is not found ')
    
if opt.save_img:
    save_dir = 'saveImg/{}_x{}_{}_YCbCr_epoch{}'.format(opt.train_dataset, opt.scale, opt.test_dataset, opt.epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
#------------------------------------------------------------------------#
# Data loader
print('===> Loading test datasets')
# data_path = join('LFData', 'test_{}_x{}.h5'.format(opt.test_dataset,opt.scale))
test_set = DatasetFromHdf5(opt)
test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False)
print('loaded {} LFIs from {}'.format(len(test_loader), opt.test_path))
an = opt.angular_num
#-------------------------------------------------------------------------#
# Build model
print("building fusion net")
model = FusionNet(opt).to(device)
print('param number: ',sum(p.numel() for p in model.parameters()))
#------------------------------------------------------------------------#

#-------------------------------------------------------------------------#    
# test
def CropPatches(image,psize,crop):
    # img [n,c,h,w]
    n, c, h, w = image.shape
    # print(image.shape)

    patch_num_h = math.ceil((h-2*crop)/psize)
    # print(patch_num_h)
    patch_num_w = math.ceil((w - 2 * crop) / psize)
    # print(patch_num_w)

    pad_h = psize * patch_num_h + 2 * crop
    pad_w = psize * patch_num_w + 2 * crop
    # print(pad_h)
    # print(pad_w)

    image_pad = torch.nn.functional.pad(image, pad=[0, pad_w-w, 0, pad_h-h], mode='constant', value=0)
    # print(image_pad.shape)
    # print(image[0, 0])
    # print(image_pad[0,0])

    patches = torch.Tensor(patch_num_h, patch_num_w, n, c, psize+2*crop, psize+2*crop).type_as(image)
    for i in range(patch_num_h):
        for j in range(patch_num_w):
            patches[i, j] = image_pad[:, :, i*psize:(i+1)*psize+2*crop, j*psize:(j+1)*psize+2*crop]

    return patches

def MergePatches(patches,h,w,psize,crop):
    # print(patches.shape)
    patch_num_h, patch_num_w, n, c = patches.shape[:4]
    pad_h = psize * patch_num_h + 2 * crop
    pad_w = psize * patch_num_w + 2 * crop

    image_pad = torch.Tensor(n, c, pad_h, pad_w)
    image_pad[:, :, :crop, :crop] = patches[0, 0, :, :, :crop, :crop]
    image_pad[:, :, :crop, -crop:] = patches[0, patch_num_w-1, :, :, :crop, -crop:]
    image_pad[:, :, -crop:, :crop] = patches[patch_num_h-1, 0, :, :, -crop:, :crop]
    image_pad[:, :, -crop:, -crop:] = patches[patch_num_h-1, patch_num_w-1, :, :, -crop:, -crop:]
    for j in range(patch_num_w):
        image_pad[:, :, :crop, j*psize+crop:(j+1)*psize+crop] = patches[0, j, :, :, :crop, crop:-crop]
        image_pad[:, :, -crop:,j*psize+crop:(j+1)*psize+crop] = patches[patch_num_h-1, j, :, :, -crop:, crop:-crop]

    for i in range(patch_num_h):
        image_pad[:, :, i*psize+crop:(i+1)*psize+crop, :crop] = patches[i, 0, :, :, crop:-crop, :crop]
        image_pad[:, :, i*psize+crop:(i+1)*psize+crop, -crop:] = patches[i, patch_num_w-1, :, :, crop:-crop, -crop:]

    for i in range(patch_num_h):
        for j in range(patch_num_w):
            image_pad[:, :, i*psize+crop:(i+1)*psize+crop, j*psize+crop:(j+1)*psize+crop] = patches[i, j, :, :, crop:-crop, crop:-crop]


    image = image_pad[:, :, :h, :w]
    return image





def crop_test(lr, hr, crop, psize):

    lr_patches = CropPatches(lr, psize=psize // opt.scale, crop=crop // opt.scale)
    hr_patches = CropPatches(hr, psize=psize, crop=crop)

    # print(lr_patches.shape)
    # print(hr_patches.shape)

    patch_num_h = lr_patches.shape[0]
    patch_num_w = lr_patches.shape[1]
    patch_crop_size = psize + 2 *crop
    an2 = lr_patches.shape[3]

    sr_fusion_patches = torch.Tensor(patch_num_h, patch_num_w, 1, an2, patch_crop_size, patch_crop_size)
    sr_copy_patches = torch.Tensor(patch_num_h, patch_num_w, 1, an2, patch_crop_size, patch_crop_size)
    sr_warp_patches = torch.Tensor(patch_num_h, patch_num_w, 1, an2, patch_crop_size, patch_crop_size)
    mask_copy_patches = torch.Tensor(patch_num_h, patch_num_w, 1, an2, patch_crop_size, patch_crop_size)
    disp_patches = torch.Tensor(patch_num_h, patch_num_w, 1, an2, patch_crop_size, patch_crop_size)

    for i in range(lr_patches.shape[0]):
        for j in range(lr_patches.shape[1]):
            sr_copy_patches[i, j], mask_copy_patches[i, j], disp_patches[i, j], sr_warp_patches[i, j], _, sr_fusion_patches[i, j] = model(lr_patches[i, j], hr_patches[i, j])

    sr_fusion = MergePatches(sr_fusion_patches, hr.shape[2], hr.shape[3], psize, crop)  # [N,an2,hs,ws]
    sr_copy = MergePatches(sr_copy_patches, hr.shape[2], hr.shape[3], psize, crop)  # [N,an2,hs,ws]
    sr_warp = MergePatches(sr_warp_patches, hr.shape[2], hr.shape[3], psize, crop)  # [N,an2,hs,ws]
    mask_copy = MergePatches(mask_copy_patches, hr.shape[2], hr.shape[3], psize, crop)
    disp = MergePatches(disp_patches, hr.shape[2], hr.shape[3], psize, crop)

    return sr_fusion, sr_copy, sr_warp, mask_copy, disp


def test():

    with torch.no_grad():
        for k,batch in enumerate(test_loader):
            # print('img ',str(k))
            # print('read batch')
            
            #------------ SR ----------------------------------------------------------------#
            lr_ycbcr, hr_ycbcr = batch[0].to(device), batch[1].to(device)

            if opt.crop:           
                crop = 96
                psize = 384

                lr_y = lr_ycbcr[:, :, 0]
                hr_y = hr_ycbcr[:, 0:1]
                sr_fusion_y, sr_copy_y, sr_warp_y, mask_copy, disp = crop_test(lr_y, hr_y, crop, psize)

                lr_cb = lr_ycbcr[:, :, 1]
                hr_cb = hr_ycbcr[:, 1:2]
                sr_fusion_cb, sr_copy_cb, sr_warp_cb, _, _ = crop_test(lr_cb, hr_cb, crop, psize)

                lr_cr = lr_ycbcr[:, :, 2]
                hr_cr = hr_ycbcr[:, 2:3]
                sr_fusion_cr, sr_copy_cr, sr_warp_cr, _, _ = crop_test(lr_cr, hr_cr, crop, psize)



                sr_fusion_ycbcr = torch.stack([sr_fusion_y, sr_fusion_cb, sr_fusion_cr], 2) #[N,an2,3,h,w]
                sr_copy_ycbcr = torch.stack([sr_copy_y, sr_copy_cb, sr_copy_cr], 2)  # [N,an2,3,h,w]
                sr_warp_ycbcr = torch.stack([sr_warp_y, sr_warp_cb, sr_warp_cr], 2)  # [N,an2,3,h,w]



            sr_fusion_ycbcr = sr_fusion_ycbcr.cpu().numpy()
            sr_copy_ycbcr = sr_copy_ycbcr.cpu().numpy()
            sr_warp_ycbcr = sr_warp_ycbcr.cpu().numpy()
                        

            
            if opt.save_img:
            
                # # mask color bar
                # cmap =  mpl.cm.plasma
                # fig, ax = plt.subplots(figsize=(1,6))
                # # fig.subplots_adjust(bottom=0.5)
                # norm = mpl.colors.Normalize(vmin=0, vmax=1)
                # fig.colorbar(
                # mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                # cax=ax,
                # orientation='vertical',
                # )
                # plt.savefig('{}/mask_colorbar.png'.formant(save_dir))
                # plt.close()
                
                
                # # disp color bar
                # cmap =  mpl.cm.summer
                # fig, ax = plt.subplots(figsize=(1,6))
                # # fig.subplots_adjust(bottom=0.5)
                # norm = mpl.colors.Normalize(vmin=-15, vmax=15)
                # fig.colorbar(
                # mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                # cax=ax,
                # orientation='vertical',
                # )
                # plt.savefig('{}/disp_colorbar.png'.format(save_dir))
                # plt.close()                
                
                for i in range(an*an):

                    img_name = '{}/SRfusion{}_view{}.png'.format(save_dir,k,i)
                    sr_sub = sr_fusion_ycbcr[0, i].transpose(1, 2, 0)
                    sr_sub = Image.fromarray((sr_sub.clip(0,1)*255.).astype('uint8'),'YCbCr')
                    img = sr_sub.convert('RGB').save(img_name)

                    img_name = '{}/SRcopy{}_view{}.png'.format(save_dir,k,i)
                    sr_sub = sr_copy_ycbcr[0, i].transpose(1, 2, 0)
                    sr_sub = Image.fromarray((sr_sub.clip(0,1)*255.).astype('uint8'),'YCbCr')
                    img = sr_sub.convert('RGB').save(img_name)

                    img_name = '{}/SRwarp{}_view{}.png'.format(save_dir,k,i)
                    sr_sub = sr_warp_ycbcr[0, i].transpose(1, 2, 0)
                    sr_sub = Image.fromarray((sr_sub.clip(0,1)*255.).astype('uint8'),'YCbCr')
                    img = sr_sub.convert('RGB').save(img_name)
  
                    cmap =  mpl.cm.plasma
                    img_name = '{}/MASKcopy{}_view{}.png'.format(save_dir,k,i)
                    plt.figure()
                    plt.figure()
                    plt.imshow(mask_copy[0,i],cmap = cmap)
                    plt.colorbar()
                    plt.savefig(img_name)
                    plt.close()
                    
                    mdic = {'mask_copy':mask_copy[0,i]}
                    mat_name = '{}/MASKcopy{}_view{}.mat'.format(save_dir, k, i)
                    io.savemat(mat_name, mdic)
                    
                    
                    img_name = '{}/Disp{}_view{}.png'.format(save_dir,k,i)
                    plt.figure()
                    plt.imshow(disp[0,i],cmap = cmap)
                    plt.colorbar()
                    plt.savefig(img_name)
                    plt.close()
                    
                    mdic = {'disp':disp[0,i]}
                    mat_name = '{}/Disp{}_view{}.mat'.format(save_dir, k, i)
                    io.savemat(mat_name, mdic)
                    


                    # cmap =  mpl.cm.plasma
                    # img_name = '{}/MASKcopy{}_view{}.png'.format(save_dir,k,i)
                    # fig = plt.gcf()
                    # plt.axis('off')
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    # plt.margins(0,0)         
                    # plt.imshow(mask_copy[0,i],cmap = cmap, vmin=0, vmax=1)
                    # plt.colorbar()
                    # plt.savefig(img_name)
                    # plt.close()
                    
                    # cmap =  mpl.cm.plasma
                    # img_name = '{}/MASKcopy{}_view{}.png'.format(save_dir,k,i)
                    # fig, ax = plt.subplots()
                    # ax.imshow(mask_copy[0,i], aspect="equal", cmap=cmap, vmin=0, vmax=1)
                    # plt.axis("off")
                    # height, width = mask_copy[0,i].shape
                    # fig.set_size_inches(width/100.0, height/100.0)
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    # plt.margins(0,0)         
                    # plt.savefig(img_name)
                    # plt.close()
                    
                    


                    
                    # cmap =  mpl.cm.summer
                    # img_name = '{}/Disp{}_view{}.png'.format(save_dir,k,i)
                    # fig, ax = plt.subplots()
                    # ax.imshow(disp[0,i], aspect="equal", cmap=cmap, vmin=-15, vmax=15)
                    # plt.axis("off")
                    # height, width = disp[0,i].shape
                    # fig.set_size_inches(width/100.0, height/100.0)
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    # plt.margins(0,0)         
                    # plt.savefig(img_name)
                    # plt.close()
                    
                    # print(tt)
            


    
#------------------------------------------------------------------------#

# for epoch in test_epochs: 
print('===> test epoch {}'.format(opt.epoch))
resume_path = join(model_dir,"model_epoch_{}.pth".format(opt.epoch))
checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['model'])        
print('loaded model {}'.format(resume_path))
model.eval()
test()
                  

