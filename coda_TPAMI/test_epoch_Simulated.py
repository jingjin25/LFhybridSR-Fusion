# test on a certain epoch to record details
# records PSNR/SSIM of SRfusion SRcopy SRwawrp for each LFI 

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os import listdir
from os.path import join
from math import ceil

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

        self.GT_y = hf.get('/GT_y')  # [N,aw,ah,h,w]
        self.LR_ycbcr = hf.get('/LR_ycbcr_{}'.format(opt.scale))  # [N,ah,aw,3,h/s,w/s]

        self.scale = opt.scale
        self.an = opt.angular_num

    def __getitem__(self, index):
        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]

        gt_y = self.GT_y[index]
        gt_y = gt_y[:self.an, :self.an, :h, :w]

        ind = ceil(self.an/2.)-1
        hr_y = gt_y[ind, ind, :, :].reshape(-1, h, w)
        gt_y = gt_y.reshape(-1, h, w)


        lr_ycbcr = self.LR_ycbcr[index]
        lr_ycbcr = lr_ycbcr[:self.an, :self.an, :, :h // self.scale, :w // self.scale].reshape(-1, 3, h // self.scale, w // self.scale)


        gt_y = torch.from_numpy(gt_y.astype(np.float32) / 255.0)
        hr_y = torch.from_numpy(hr_y.astype(np.float32) / 255.0)
        lr_ycbcr = torch.from_numpy(lr_ycbcr.astype(np.float32) / 255.0)

        return gt_y, lr_ycbcr, hr_y

    def __len__(self):
        return self.GT_y.shape[0]
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
    save_dir = 'saveImg/{}_x{}_{}_epoch{}'.format(opt.train_dataset, opt.scale, opt.test_dataset, opt.epoch)
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
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16. / 255.
    rgb[:,1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)
    
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


def compt_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mask_norm(mask1,mask2):
    softmax = torch.nn.Softmax(dim=1)
    mask = torch.stack((mask1,mask2),1)
    mask = softmax(mask)
    mask1 = mask[:,0]
    mask2 = mask[:,1]
    return mask1, mask2


def test():

    avg_psnr_y = 0
    avg_psnr_y_copy = 0
    avg_psnr_y_warp = 0
    
    avg_ssim_y = 0
    avg_ssim_y_copy = 0
    avg_ssim_y_warp = 0
    
    lf_list = []                
    lf_psnr_list = []
    lf_psnr_copy_list = []
    lf_psnr_warp_list= []                
    lf_ssim_list = []
    lf_ssim_copy_list = []
    lf_ssim_warp_list = []    
    
    
    with torch.no_grad():
        for k,batch in enumerate(test_loader):
            # print('img ',str(k))
            # print('read batch')
            
            #------------ SR ----------------------------------------------------------------#
            gt_y, lr_ycbcr, hr = batch[0].numpy(), batch[1].to(device), batch[2].to(device)
            lr = lr_ycbcr[:, :, 0]

            if opt.crop:           
                crop = 16
                psize = 96
                
                lr_patches = CropPatches(lr, psize=psize // opt.scale, crop=crop // opt.scale)
                hr_patches = CropPatches(hr, psize=psize, crop=crop)

                sr_fusion_patches = torch.Tensor(lr_patches.shape[0], lr_patches.shape[1], 1, an*an, hr_patches.shape[4], hr_patches.shape[5])
                sr_copy_patches = torch.Tensor(lr_patches.shape[0], lr_patches.shape[1], 1, an*an, hr_patches.shape[4], hr_patches.shape[5])
                sr_warp_patches = torch.Tensor(lr_patches.shape[0], lr_patches.shape[1], 1, an*an, hr_patches.shape[4], hr_patches.shape[5])
                mask_copy_patches = torch.Tensor(lr_patches.shape[0], lr_patches.shape[1], 1, an*an, hr_patches.shape[4], hr_patches.shape[5])
                disp_patches = torch.Tensor(lr_patches.shape[0], lr_patches.shape[1], 1, an*an, hr_patches.shape[4], hr_patches.shape[5])

                for i in range(lr_patches.shape[0]):
                    for j in range(lr_patches.shape[1]):
                         sr_copy_patches[i, j], mask_copy_patches[i, j], disp_patches[i, j], sr_warp_patches[i, j], _, sr_fusion_patches[i, j] = model(lr_patches[i, j],hr_patches[i, j])

                sr_fusion = MergePatches(sr_fusion_patches, hr.shape[2], hr.shape[3], psize, crop)  #[N,an2,hs,ws]
                sr_copy = MergePatches(sr_copy_patches, hr.shape[2], hr.shape[3], psize, crop)  #[N,an2,hs,ws]
                sr_warp = MergePatches(sr_warp_patches, hr.shape[2], hr.shape[3], psize, crop)  #[N,an2,hs,ws]
                mask_copy = MergePatches(mask_copy_patches, hr.shape[2], hr.shape[3], psize, crop)  # [N,an2,hs,ws]
                disp = MergePatches(disp_patches, hr.shape[2], hr.shape[3], psize, crop)

            else:
                sr_copy, mask_copy_norm, disp, sr_warp, mask_warp_norm, sr_fusion = model(lr, hr)


            N,an2,_,h,w = lr_ycbcr.shape
            sr_ycbcr = torch.nn.functional.interpolate(lr_ycbcr.view(N*an2, 3, h, w), scale_factor=opt.scale, mode='bicubic', align_corners=False).view(N,an2,3,h*opt.scale,w*opt.scale)

            sr_copy_ycbcr = sr_ycbcr.clone()
            sr_warp_ycbcr = sr_ycbcr.clone()
            sr_fusion_ycbcr = sr_ycbcr.clone()


            sr_fusion_ycbcr[:, :, 0] = sr_fusion
            sr_copy_ycbcr[:, :, 0] = sr_copy
            sr_warp_ycbcr[:, :, 0] = sr_warp

            sr_fusion_ycbcr = sr_fusion_ycbcr.cpu().numpy()
            sr_copy_ycbcr = sr_copy_ycbcr.cpu().numpy()
            sr_warp_ycbcr = sr_warp_ycbcr.cpu().numpy()

            sr_fusion_y = sr_fusion.cpu().numpy()
            sr_copy_y = sr_copy.cpu().numpy()
            sr_warp_y = sr_warp.cpu().numpy()

            mask_copy = mask_copy.cpu().numpy()
            disp = disp.cpu().numpy()
                        
            #--------------------------------------compute average PSNR/SSIM for this LFI -----------------------------------#                       
            lf_psnr_y = 0
            lf_psnr_y_copy = 0
            lf_psnr_y_warp = 0            
            lf_ssim_y = 0
            lf_ssim_y_copy = 0
            lf_ssim_y_warp = 0            
            for i in range(an*an):
                if opt.save_img:
                    # img_name = '{}/SRcopy{}_view{}.png'.format(save_dir,k,i)
                    # sr_copy_rgb_temp = ycbcr2rgb(np.transpose(sr_copy_ycbcr[0,i],(1,2,0)))
                    # img = (sr_copy_rgb_temp.clip(0,1)*255.0).astype(np.uint8)
                    # # misc.imsave(img_name,img)
                    # Image.fromarray(img).convert('RGB').save(img_name)
                    #
                    # img_name = '{}/SRwarp{}_view{}.png'.format(save_dir,k,i)
                    # sr_warp_rgb_temp = ycbcr2rgb(np.transpose(sr_warp_ycbcr[0,i],(1,2,0)))
                    # img = (sr_warp_rgb_temp.clip(0,1)*255.0).astype(np.uint8)
                    # Image.fromarray(img).convert('RGB').save(img_name)
                    
                    img_name = '{}/SRfusion{}_view{}.png'.format(save_dir,k,i)  
                    sr_fusion_rgb_temp = ycbcr2rgb(np.transpose(sr_fusion_ycbcr[0,i],(1,2,0)))
                    img = (sr_fusion_rgb_temp.clip(0,1)*255.0).astype(np.uint8) 
                    Image.fromarray(img).convert('RGB').save(img_name)
                        
                    # cmap =  mpl.cm.plasma
                    # img_name = '{}/MASKcopy{}_view{}.png'.format(save_dir,k,i)
                    # plt.figure()
                    # plt.imshow(mask_copy[0,i],cmap = cmap)
                    # plt.colorbar()
                    # plt.savefig(img_name)
                    # plt.close()
                    # img_name = '{}/MASKwarp{}_view{}.png'.format(save_dir,k,i)
                    # plt.figure()
                    # plt.imshow(mask_warp[0,i],cmap = cmap)
                    # plt.colorbar()
                    # plt.savefig(img_name)
                    # plt.close()

                    # img_name = '{}/Disp{}_view{}.png'.format(save_dir,k,i)
                    # plt.figure()
                    # plt.imshow(disp[0,i],cmap = cmap)
                    # plt.colorbar()
                    # plt.savefig(img_name)
                    # plt.close()

            
                if i != (an+1) * ( math.ceil(an/2)-1):                
                    lf_psnr_y += compt_psnr(gt_y[0,i],sr_fusion_y[0,i]) / (an*an-1 )
                    lf_psnr_y_copy += compt_psnr(gt_y[0,i],sr_copy_y[0,i]) / (an*an-1 )
                    lf_psnr_y_warp += compt_psnr(gt_y[0,i],sr_warp_y[0,i]) / (an*an-1 )

                    lf_ssim_y += compare_ssim(gt_y[0,i],sr_fusion_y[0,i]) / (an*an-1 )
                    lf_ssim_y_copy += compare_ssim(gt_y[0,i],sr_copy_y[0,i]) / (an*an-1 )
                    lf_ssim_y_warp += compare_ssim(gt_y[0,i],sr_warp_y[0,i]) / (an*an-1 )

                    
            lf_list.append(k)
            lf_psnr_list.append(lf_psnr_y)
            lf_psnr_copy_list.append(lf_psnr_y_copy)
            lf_psnr_warp_list.append(lf_psnr_y_warp)
            lf_ssim_list.append(lf_ssim_y)
            lf_ssim_copy_list.append(lf_ssim_y_copy)
            lf_ssim_warp_list.append(lf_ssim_y_warp)
    
    csv_name = 'quan_results/res_{}_{}_x{}_Epoch{}.csv'.format(opt.train_dataset,opt.test_dataset,opt.scale, opt.epoch)
    dataframe_lfi = pd.DataFrame({'lfiNo': lf_list, 'psnr Y fusion':lf_psnr_list, 'psnr Y copy':lf_psnr_copy_list, 'psnr Y warp':lf_psnr_warp_list, 'ssim Y fusion':lf_ssim_list, 'ssim Y copy':lf_ssim_copy_list, 'ssim Y warp':lf_ssim_warp_list})                  
    dataframe_lfi.to_csv(csv_name,index = False,sep=',',mode='a') 

    dataframe_lfi = pd.DataFrame({'summary': ['avg'], 'psnr Y fusion':[np.mean(lf_psnr_list)], 'psnr Y copy':[np.mean(lf_psnr_copy_list)], 'psnr Y warp':[np.mean(lf_psnr_warp_list)], 'ssim Y fusion':[np.mean(lf_ssim_list)], 'ssim Y copy':[np.mean(lf_ssim_copy_list)], 'ssim Y warp':[np.mean(lf_ssim_warp_list)]})                  
    dataframe_lfi.to_csv(csv_name,index = False,sep=',',mode='a')

        


                

    
#------------------------------------------------------------------------#

# for epoch in test_epochs: 
print('===> test epoch {}'.format(opt.epoch))
resume_path = join(model_dir,"model_epoch_{}.pth".format(opt.epoch))
checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['model'])        
print('loaded model {}'.format(resume_path))
model.eval()
test()
                  

