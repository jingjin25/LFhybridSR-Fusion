
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy
from scipy import misc

from Model_SRNet import SRNet
from Model_WarpNet import WarpNet
    

class FusionNet(nn.Module):
    
    def __init__(self, opt):
        
        super(FusionNet, self).__init__()
        
        self.srnet = SRNet(opt)
        self.warpnet = WarpNet(opt)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, lr, hr):
            
        sr_copy, mask_copy = self.srnet(lr, hr)
        disp, sr_warp, mask_warp = self.warpnet(lr, hr)

        mask = torch.stack((mask_copy, mask_warp), 1) #[N,2,an2,h,w]
        mask = self.softmax(mask)
        mask_copy_norm = mask[:, 0]  #[N,an2,h,w]
        mask_warp_norm = mask[:, 1]

        sr_fusion = sr_copy * mask_copy_norm + sr_warp * mask_warp_norm
               
        return sr_copy, mask_copy_norm, disp, sr_warp, mask_warp_norm, sr_fusion