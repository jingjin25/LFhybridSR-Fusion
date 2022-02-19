
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy
from scipy import misc

from hybridSR_copy import copyNet_4x, copyNet_8x
from hybridSR_warp import warpNet
    
class fusionNet_4x(nn.Module):
    
    def __init__(self, an, scale, copy_layer, warp_layer):
        
        super(fusionNet_4x, self).__init__()
            
        self.copynet = copyNet_4x(an, copy_layer)
        self.warpnet = warpNet(an, scale, warp_layer)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, lr, hr, hr_2):
            
        sr_copy_2x, sr_copy_4x, mask_copy = self.copynet(lr, hr, hr_2)
        disp, sr_warp, mask_warp = self.warpnet(lr, hr)
        
        mask = torch.stack((mask_copy,mask_warp),1) #[N,2,81,h,w]
        mask = self.softmax(mask)
        mask_copy_norm = mask[:,0]  # [N,81,h,w]
        mask_warp_norm = mask[:,1]
                
        sr_fusion = sr_copy_4x * mask_copy_norm + sr_warp * mask_warp_norm
               
        return sr_copy_2x, sr_copy_4x, mask_copy, disp, sr_warp, mask_warp, sr_fusion
 
 
 
class fusionNet_8x(nn.Module):
    
    def __init__(self, an, scale, copy_layer, warp_layer):
        
        super(fusionNet_8x, self).__init__()
        
        self.copynet = copyNet_8x(an, copy_layer)
        self.warpnet = warpNet(an, scale, warp_layer)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, lr, hr, hr_2, hr_4):
            
        sr_copy_2x, sr_copy_4x, sr_copy_8x, mask_copy = self.copynet(lr, hr, hr_2, hr_4)
        disp, sr_warp, mask_warp = self.warpnet(lr, hr)
        
        mask = torch.stack((mask_copy,mask_warp),1) #[N,2,81,h,w]
        mask = self.softmax(mask)
        mask_copy_norm = mask[:,0]  # [N,81,h,w]
        mask_warp_norm = mask[:,1]
                
        sr_fusion = sr_copy_8x * mask_copy_norm + sr_warp * mask_warp_norm
               
        return sr_copy_2x, sr_copy_4x, sr_copy_8x, mask_copy, disp, sr_warp, mask_warp, sr_fusion

 