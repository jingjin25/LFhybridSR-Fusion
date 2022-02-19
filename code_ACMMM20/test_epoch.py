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

from hybridSR_fusion import fusionNet_4x, fusionNet_8x


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
parser.add_argument("--angular_num", type=int, default=9, help="Size of angular dim")
parser.add_argument("--copy_layer", type=int, default=6, help="layer number for copyNet")
parser.add_argument("--warp_layer", type=int, default=3, help="layer number for warpNet")

parser.add_argument("--epoch", type=int, default=1,help="start epoch for test")

parser.add_argument("--save_img", type=int, default=0,help="save image or not")
# parser.add_argument("--lfi_info", type=int, default=1,help="save results for each testing LF image")

parser.add_argument("--crop", type=int, default=0,help="crop the image into patches when not enough memory")
opt = parser.parse_args()
print(opt)
#-----------------------------------------------------------------------------------#   
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, scale, an):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        
        self.GT_rgb = hf.get('/GT_rgb')  #[N,ah,aw,3,h,w]            
        self.GT_y = hf.get('/GT_y')      #[N,aw,ah,h,w]
        self.LR_ycbcr_up = hf.get('/LR_up_ycbcr') #[N,ah,aw,3,h,w]
        self.LR = hf.get('/LR') #[N,ah,aw,h/s,w/s]     
        self.HR = hf.get('/HR') #[N,1,1,h,w]
        self.HR_2 = hf.get('/HR_2') 
        self.HR_4 = hf.get('/HR_4')
        
        self.scale = scale
        self.an = an

    def __getitem__(self, index):

        h = self.GT_y.shape[3]
        w = self.GT_y.shape[4]
        
        gt_rgb = self.GT_rgb[index]
        gt_rgb = gt_rgb[:self.an,:self.an,:,:h,:w].reshape(-1,3,h,w)
        gt_y = self.GT_y[index]
        gt_y = gt_y[:self.an,:self.an,:h,:w].reshape(-1,h,w)
        lr_ycbcr_up = self.LR_ycbcr_up[index]
        lr_ycbcr_up = lr_ycbcr_up[:self.an,:self.an,:,:h,:w].reshape(-1,3,h,w)
        lr = self.LR[index]
        lr = lr[:self.an,:self.an,:h//self.scale,:w//self.scale].reshape(-1,h//self.scale,w//self.scale)
        hr = self.HR[index]
        hr = hr[:self.an,:self.an,:h,:w].reshape(-1,h,w)
        hr_2 = self.HR_2[index]
        hr_2 = hr_2[:self.an,:self.an,:h//2,:w//2].reshape(-1,h//2,w//2)
        hr_4 = self.HR_4[index]
        hr_4 = hr_4[:self.an,:self.an,:h//4,:w//4].reshape(-1,h//4,w//4)
        
        gt_rgb = torch.from_numpy(gt_rgb.astype(np.float32)/255.0)
        gt_y   = torch.from_numpy(gt_y.astype(np.float32)/255.0)
        lr_ycbcr_up = torch.from_numpy(lr_ycbcr_up.astype(np.float32)/255.0)
        lr = torch.from_numpy(lr.astype(np.float32)/255.0) 
        hr = torch.from_numpy(hr.astype(np.float32)/255.0)
        hr_2 = torch.from_numpy(hr_2.astype(np.float32)/255.0)
        hr_4 = torch.from_numpy(hr_4.astype(np.float32)/255.0)

        return gt_rgb, gt_y, lr_ycbcr_up, lr, hr, hr_2, hr_4
        
    def __len__(self):
        return self.GT_rgb.shape[0]
#-----------------------------------------------------------------------------------#        

#-------------------------------------------------------------------------------#
if opt.model_dir == "":
    model_dir = 'model_x{}_{}_lr{}_step{}x{}_cL{}_wL{}'.format(opt.scale,opt.train_dataset,opt.lr,opt.step,opt.reduce,opt.copy_layer,opt.warp_layer)
else:
    model_dir = opt.model_dir
    
if not os.path.exists(model_dir):
    print('model folder is not found ')
    
if opt.save_img:
    save_dir = 'saveImg/{}_x{}_{}'.format(opt.train_dataset, opt.scale, opt.test_dataset )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
#------------------------------------------------------------------------#
# Data loader
print('===> Loading test datasets')
data_path = join('LFData', 'test_{}_x{}.h5'.format(opt.test_dataset,opt.scale))
test_set = DatasetFromHdf5(data_path,opt.scale,opt.angular_num)
test_loader = DataLoader(dataset=test_set,batch_size=1,shuffle=False)
print('loaded {} LFIs from {}'.format(len(test_loader),data_path))
an = opt.angular_num
#-------------------------------------------------------------------------#
# Build model
print("building fusion net")
if opt.scale == 4 :
    model = fusionNet_4x(an,opt.scale,opt.copy_layer,opt.warp_layer).to(device)
elif opt.scale == 8:
    model = fusionNet_8x(an,opt.scale,opt.copy_layer,opt.warp_layer).to(device)
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
    
def CropPatches(image,len,crop):
    #left [1,an2,h,lw]
    #middles[n,an2,h,mw]
    #right [1,an2,h,rw]
    an,h,w = image.shape[1:4]
    left = image[:,:,:,0:len+crop]
    num = math.floor((w-len-crop)/len)
    middles = torch.Tensor(num,an,h,len+crop*2).to(image.device)
    for i in range(num):
        middles[i] = image[0,:,:,(i+1)*len-crop:(i+2)*len+crop]      
    right = image[:,:,:,-(len+crop):]
    return left,middles,right

def MergePatches(left,middles,right,h,w,len,crop):

    n,a = left.shape[0:2]
    out = torch.Tensor(n,a,h,w).to(left.device)
    out[:,:,:,:len] = left[:,:,:,:-crop]
    for i in range(middles.shape[0]): 
        out[:,:,:,len*(i+1):len*(i+2)] = middles[i:i+1,:,:,crop:-crop]        
    out[:,:,:,-len:]=right[:,:,:,crop:]
    return out

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

def model_test(lr, hr, hr_2,hr_4):   

    if opt.scale == 4: 
        _,sr_copy, mask_copy, disp, sr_warp, mask_warp, sr_fusion = model(lr, hr, hr_2)
    elif opt.scale == 8:
        _,_,sr_copy, mask_copy, disp, sr_warp, mask_warp, sr_fusion = model(lr, hr, hr_2, hr_4)
        
    mask_copy, mask_warp = mask_norm(mask_copy,mask_warp)
    
    sr_copy = sr_copy.cpu()
    mask_copy = mask_copy.cpu()
    disp = disp.cpu()
    sr_warp = sr_warp.cpu()
    mask_warp = mask_warp.cpu()
    sr_fusion = sr_fusion.cpu()
    
    return sr_copy, sr_warp, sr_fusion, mask_copy, mask_warp

    
def test(epoch):

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
            
            #---------------------------------------------------- SR ----------------------------------------------------------------#
            gt_rgb, gt_y, gt_ycbcr, lr, hr, hr_2, hr_4 = batch[0].numpy(),batch[1].numpy(),batch[2].numpy(),batch[3],batch[4],batch[5],batch[6]
            
            sr_copy_ycbcr = gt_ycbcr.copy()
            sr_warp_ycbcr = gt_ycbcr.copy()
            sr_fusion_ycbcr = gt_ycbcr.copy()
                     
                    
            if opt.crop:           
                crop = 8
                length = 64
                
                lr_l, lr_m, lr_r = CropPatches(lr,length//opt.scale,crop//opt.scale) 
                hr_l, hr_m, hr_r = CropPatches(hr,length,crop) 
                hr_2_l, hr_2_m, hr_2_r = CropPatches(hr_2,length//2,crop//2)                                           
                hr_4_l, hr_4_m, hr_4_r = CropPatches(hr_4,length//4,crop//4)
                
                lr_l = lr_l.to(device)
                hr_l = hr_l.to(device)
                hr_2_l = hr_2_l.to(device) 
                hr_4_l = hr_4_l.to(device)                
                sr_copy_l, sr_warp_l, sr_fusion_l, mask_copy_l, mask_warp_l = model_test(lr_l, hr_l, hr_2_l,hr_4_l)    

                
                # print('middles to gpu')   
                # disp_m = torch.Tensor(lr_m.shape[0], an*an, hr_m.shape[2], hr_m.shape[3])                
                sr_copy_m = torch.Tensor(lr_m.shape[0], an*an, hr_m.shape[2], hr_m.shape[3])
                mask_copy_m = torch.Tensor(lr_m.shape[0], an*an, hr_m.shape[2], hr_m.shape[3])
                sr_warp_m = torch.Tensor(lr_m.shape[0], an*an, hr_m.shape[2], hr_m.shape[3])
                mask_warp_m = torch.Tensor(lr_m.shape[0], an*an, hr_m.shape[2], hr_m.shape[3])
                sr_fusion_m = torch.Tensor(lr_m.shape[0], an*an, hr_m.shape[2], hr_m.shape[3])
                for i in range(lr_m.shape[0]):
                    lr_m_t = lr_m[i:i+1].to(device)
                    hr_m_t = hr_m[i:i+1].to(device)
                    hr_2_m_t = hr_2_m[i:i+1].to(device) 
                    hr_4_m_t = hr_4_m[i:i+1].to(device) 
                    sr_copy_m[i:i+1], sr_warp_m[i:i+1], sr_fusion_m[i:i+1], mask_copy_m[i:i+1], mask_warp_m[i:i+1] = model_test(lr_m_t,hr_m_t,hr_2_m_t,hr_4_m_t)   


                lr_r = lr_r.to(device)
                hr_r = hr_r.to(device)
                hr_2_r = hr_2_r.to(device)
                hr_4_r = hr_4_r.to(device)     
                sr_copy_r, sr_warp_r, sr_fusion_r, mask_copy_r, mask_warp_r = model_test(lr_r,hr_r,hr_2_r,hr_4_r)

            
                sr_copy = MergePatches(sr_copy_l, sr_copy_m, sr_copy_r, hr.shape[2], hr.shape[3], length, crop)  #[N,an2,hs,ws]     
                mask_copy = MergePatches(mask_copy_l, mask_copy_m, mask_copy_r, hr.shape[2], hr.shape[3], length, crop)  #[N,an2,hs,ws] 
                sr_warp = MergePatches(sr_warp_l, sr_warp_m, sr_warp_r, hr.shape[2], hr.shape[3], length, crop)  #[N,an2,hs,ws] 
                mask_warp = MergePatches(mask_warp_l, mask_warp_m, mask_warp_r, hr.shape[2], hr.shape[3], length, crop)  #[N,an2,hs,ws] 
                sr_fusion = MergePatches(sr_fusion_l, sr_fusion_m, sr_fusion_r, hr.shape[2], hr.shape[3], length, crop)  #[N,an2,hs,ws]
                mask_copy,mask_warp = mask_norm(mask_copy,mask_warp)
                
            else:            
                lr = lr.to(device)
                hr = hr.to(device)
                hr_2 = hr_2.to(device)          
                hr_4 = hr_4.to(device)
                sr_copy, sr_warp, sr_fusion, mask_copy, mask_warp = model(lr,hr,hr_2,hr_4)
            

            sr_copy_y = sr_copy.numpy()
            mask_copy = mask_copy.numpy()
            sr_warp_y = sr_warp.numpy()
            mask_warp = mask_warp.numpy()
            sr_fusion_y = sr_fusion.numpy()
            
            sr_copy_ycbcr[:,:,0] = sr_copy_y
            sr_warp_ycbcr[:,:,0] = sr_warp_y
            sr_fusion_ycbcr[:,:,0] = sr_fusion_y
                        
            #--------------------------------------compute average PSNR/SSIM for this LFI -----------------------------------#                       
            lf_psnr_y = 0
            lf_psnr_y_copy = 0
            lf_psnr_y_warp = 0            
            lf_ssim_y = 0
            lf_ssim_y_copy = 0
            lf_ssim_y_warp = 0            
            for i in range(an*an):
                if opt.save_img:
                    img_name = '{}/SRcopy{}_view{}.png'.format(save_dir,k,i) 
                    sr_copy_rgb_temp = ycbcr2rgb(np.transpose(sr_copy_ycbcr[0,i],(1,2,0)))
                    img = (sr_copy_rgb_temp.clip(0,1)*255.0).astype(np.uint8) 
                    # misc.imsave(img_name,img)
                    Image.fromarray(img).convert('RGB').save(img_name)

                    img_name = '{}/SRwarp{}_view{}.png'.format(save_dir,k,i)
                    sr_warp_rgb_temp = ycbcr2rgb(np.transpose(sr_warp_ycbcr[0,i],(1,2,0)))
                    img = (sr_warp_rgb_temp.clip(0,1)*255.0).astype(np.uint8) 
                    Image.fromarray(img).convert('RGB').save(img_name)
                    
                    img_name = '{}/SRfusion{}_view{}.png'.format(save_dir,k,i)  
                    sr_fusion_rgb_temp = ycbcr2rgb(np.transpose(sr_fusion_ycbcr[0,i],(1,2,0)))
                    img = (sr_fusion_rgb_temp.clip(0,1)*255.0).astype(np.uint8) 
                    Image.fromarray(img).convert('RGB').save(img_name)
                        
                    cmap =  mpl.cm.plasma
                    img_name = '{}/MASKcopy{}_view{}.png'.format(save_dir,k,i)
                    plt.figure()
                    plt.imshow(mask_copy[0,i],cmap = cmap)
                    plt.colorbar()
                    plt.savefig(img_name)
                    plt.close()
                    img_name = '{}/MASKwarp{}_view{}.png'.format(save_dir,k,i)
                    plt.figure()
                    plt.imshow(mask_warp[0,i],cmap = cmap)
                    plt.colorbar()
                    plt.savefig(img_name)   
                    plt.close()            
            
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
    
    csv_name = 'quan_results/res_{}_Epoch{}_{}_x{}.csv'.format(opt.train_dataset,epoch,opt.test_dataset,opt.scale)
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
test(opt.epoch)
                  

