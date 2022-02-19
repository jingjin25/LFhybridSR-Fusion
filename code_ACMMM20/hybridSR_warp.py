
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from torch.autograd import Variable
from math import floor

import numpy as np
from model_utils import warp
from scipy import misc

class DepthBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthBlock, self).__init__()    
        
        self.conv1 = nn.Conv2d(in_channels = in_ch, out_channels = out_ch,kernel_size = 3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels = out_ch, out_channels = out_ch,kernel_size = 3,stride=1,padding=1)
        self.relu = nn.ReLU(inplace=True)        
        self.bn = nn.BatchNorm2d(out_ch)        
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        return out           
    

class warpNet(nn.Module):
    
    def __init__(self, an, scale, warp_layer):
        
        super(warpNet, self).__init__()
        
        self.an = an
        self.an2 = an*an
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1_h = nn.Conv2d(in_channels=self.an, out_channels=self.an*10, kernel_size=3, stride=1, padding=1)
        self.conv1_v = nn.Conv2d(in_channels=self.an, out_channels=self.an*10, kernel_size=3, stride=1, padding=1)
        
        self.mlt_h = self.make_layer(layer_num=warp_layer, in_ch=self.an*10, out_ch=self.an*10)
        self.mlt_v = self.make_layer(layer_num=warp_layer, in_ch=self.an*10, out_ch=self.an*10)
        
        self.conv2_h = nn.Conv2d(in_channels=self.an*10, out_channels=self.an, kernel_size=3, stride=1, padding=1)
        self.conv2_v = nn.Conv2d(in_channels=self.an*10, out_channels=self.an, kernel_size=3, stride=1, padding=1)
        
        self.conv2_h_mask = nn.Conv2d(in_channels=self.an*10, out_channels=self.an, kernel_size=3, stride=1, padding=1)
        self.conv2_v_mask = nn.Conv2d(in_channels=self.an*10, out_channels=self.an, kernel_size=3, stride=1, padding=1)
                       
        self.merge = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.merge_mask = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        
        if scale == 2:
            self.convt = nn.ConvTranspose2d(in_channels = self.an2,out_channels = self.an2,kernel_size = 4,stride = 2,padding=1)        
        elif scale == 4:
            self.convt = nn.Sequential(
                nn.ConvTranspose2d(in_channels = self.an2,out_channels = self.an2,kernel_size=4,stride=2,padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels = self.an2,out_channels = self.an2,kernel_size=4,stride=2,padding=1),
                )
        elif scale == 8:
            self.convt = nn.Sequential(
                nn.ConvTranspose2d(in_channels = self.an2,out_channels = self.an2,kernel_size=4,stride=2,padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels = self.an2,out_channels = self.an2,kernel_size=4,stride=2,padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels = self.an2,out_channels = self.an2,kernel_size=4,stride=2,padding=1),                
                )
                
                        
     
    def make_layer(self,layer_num,in_ch,out_ch):
        layers = []
        for i in range(layer_num):
            layers.append( DepthBlock(in_ch,out_ch))
        return nn.Sequential(*layers)     

    def forward(self, lr, hr):

        N,_,h,w = lr.shape #lr [N,81,h,w]  
        _,_,hs,ws = hr.shape #hr [N,1,hs,ws]      
             
       # disparity estimation
        # horizontal       
        stack_h = lr.view(N,self.an,self.an,h,w).view(N*self.an,self.an,h,w) #[ah,aw,h,w]
        feat_h = self.relu(self.conv1_h(stack_h))
        feat_h = self.mlt_h(feat_h) #[ah,fn,h,w]
        #print('feat_h',feat_h.shape)
        disp_h = self.conv2_h(feat_h).view(-1,h,w) #[ah,aw,h,w]->[N*ah*aw,h,w]
        mask_h = self.conv2_h_mask(feat_h).view(-1,h,w)
        #print('disp_h',disp_h.shape)
        # vertical
        stack_v = lr.view(N,self.an,self.an,h,w).transpose(1,2).contiguous().view(N*self.an,self.an,h,w) #[aw,ah,h,w]
        feat_v = self.relu(self.conv1_v(stack_v))
        feat_v = self.mlt_v(feat_v) #[aw,fn,h,w]
        #print('feat_v',feat_v.shape)
        disp_v = self.conv2_v(feat_v).transpose(0,1).contiguous().view(-1,h,w) #[aw,ah,h,w]->[ah,aw,h,w]->[N*ah*aw,h,w]
        mask_v = self.conv2_v_mask(feat_v).transpose(0,1).contiguous().view(-1,h,w)
 
        # merge
        disp = torch.stack((disp_h,disp_v),1)  #[N*81,2,h,w]
        disp = self.merge(disp).view(-1,self.an2,h,w) #[N*81,1,h,w]->[N,81,h,w] 
        
        mask = torch.stack((mask_h,mask_v),1)
        mask = self.merge_mask(mask).view(-1,self.an2,h,w)
               
        # upsample
        disp = self.convt(disp)  # [N,81,hs,ws]
        mask = self.convt(mask)
        
        
        # inverse warping
        warpedHR = Variable(torch.Tensor(N,self.an2,hs,ws),requires_grad=True).type_as(lr)  #[N,81,hs,ws]
        for i in range(self.an2):
            warpedHR[:,i:i+1,:,:] = warp(disp[:,i,:,:], hr, i, self.an)  

               
        return disp,warpedHR,mask
 


 