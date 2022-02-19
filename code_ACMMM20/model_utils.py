
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import copy
from torch.autograd import Variable
from math import floor,ceil


def warp(disp, HR, index, an):

    # disparity LF [N,h,w]
    # HR RGB  [N,1,h,w] 
    # index 0-(an2-1)
    # an number of angular dim    

    N,_,h,w = HR.shape
    index_h = floor( index / an ) - (ceil(an/2)-1)
    index_w = index % an - (ceil(an/2)-1)
    
    # generate grid
    XX = Variable(torch.arange(0,w).view(1,1,w).expand(N,h,w)).type_as(disp) #[N,h,w]
    YY = Variable(torch.arange(0,h).view(1,h,1).expand(N,h,w)).type_as(disp)
    
    grid_w = XX + disp*index_w
    grid_h = YY + disp*index_h
    
    grid_w_norm = 2.0 * grid_w / (w-1) -1.0
    grid_h_norm = 2.0 * grid_h / (h-1) -1.0
            
    grid = torch.stack((grid_w_norm, grid_h_norm),dim=3) #[N,h,w,2]
    
    # inverse warp
    warped_HR = functional.grid_sample(HR,grid) # [N,1,h,w] wared HR
    
    return warped_HR

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
    num = floor((w-len-crop)/len)
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
    
