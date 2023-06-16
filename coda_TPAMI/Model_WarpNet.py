
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from torch.autograd import Variable
from math import floor
import math

import numpy as np
from model_utils import warp
from scipy import misc


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        # self.pool4 = nn.MaxPool2d(2)
        # self.conv5 = DoubleConv(512, 1024)

        # self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.conv10_m = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        # p4 = self.pool4(c4)
        # c5 = self.conv5(p4)
        # up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6 = self.conv6(merge6)
        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        c10_m = self.conv10_m(c9)
        return c10, c10_m


class Refine_Dense(nn.Module):
    def __init__(self, in_ch, grow_ch, out_ch, layer_num):
        super(Refine_Dense, self).__init__()

        self.bottlenect = nn.ModuleList([
            nn.Conv2d(in_ch+i*grow_ch, grow_ch, 1, 1, 0, bias=True) for i in range(layer_num)
        ])
        self.transm = nn.Conv2d(in_ch + layer_num * grow_ch, out_ch, 1, 1, 0, bias=True)
        # self.transm_mask = nn.Conv2d(in_ch + layer_num * grow_ch, out_ch, 1, 1, 0, bias=True)

        self.convs = nn.ModuleList([
            nn.Conv2d(grow_ch, grow_ch, 3, 1, 1, bias=True) for i in range(layer_num)
        ])
        self.layer_num = layer_num
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for i in range(self.layer_num):
            x_comp = self.relu(self.bottlenect[i](x))
            x_out = self.relu(self.convs[i](x_comp))
            x = torch.cat([x, x_out], 1)
        out = self.transm(x)
        # mask = self.transm_mask(x)
        return out


class HR_fea(nn.Module):
    def __init__(self, in_ch, out_ch, layer):
        super(HR_fea, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=True)
        block = []
        for _ in range(layer):
            block.append(nn.Conv2d(64, 64, 3, 1, 1, bias=True))
            block.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*block)
        self.conv1 = nn.Conv2d(64, out_ch, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv_block(out)
        out = self.conv1(out)
        return out


class DepthNet(nn.Module):
    def __init__(self, an, scale):
        super(DepthNet, self).__init__()
        
        self.unet = UNet(in_ch=an*an-1, out_ch=an*an)

        up = []
        for _ in range(int(math.log(scale, 2))):
            up.append(nn.ReLU(inplace=True))
            up.append(nn.Conv2d(an*an, 4*an*an, 3, 1, 1, bias=True))
            up.append(nn.PixelShuffle(2))
        self.upsampler = nn.Sequential(*up)

        up_mask = []
        for _ in range(int(math.log(scale, 2))):
            up_mask.append(nn.ReLU(inplace=True))
            up_mask.append(nn.Conv2d(an*an, 4*an*an, 3, 1, 1, bias=True))
            up_mask.append(nn.PixelShuffle(2))
        self.upsampler_mask = nn.Sequential(*up_mask)

        self.disp_conv_fea = nn.Conv2d(1, 64, 3, 1, 1, bias=True)

        self.hr_conv = HR_fea(in_ch=1, out_ch=64, layer=6)
        self.refine_res = Refine_Dense(in_ch=64+64, grow_ch=16, out_ch=1, layer_num=6)

        self.relu = nn.ReLU(inplace=True)        

        
    def forward(self, lr, hr):

        # lr [N,an2,h,w]
        # hr [N,1,sh,sw]
        N, an2, h, w = lr.shape
        _, _, sh, sw = hr.shape
        an = int(math.sqrt(an2))
        ind_cv = torch.tensor(an // 2 * an + an // 2)

        lr_side = torch.cat([lr[:,:ind_cv], lr[:,ind_cv+1:]],1) # [N,an2-1,h,w]
        # print('lr side', lr_side.shape)

        disp_init, mask = self.unet(lr_side)  # [N,an2,h,w]

        disp_init = self.upsampler(disp_init) # [N,an2,sh,sw]
        mask = self.upsampler_mask(mask)

        disp_init_fea = self.disp_conv_fea(disp_init.view(N*an2, 1, sh, sw)) #[N*an2,64,sh,sw]


        hr_fea = self.hr_conv(hr.view(N,-1,sh,sw)) #[N,64,sh,sw]
        hr_fea = hr_fea.repeat(an2, 1, 1, 1) #[N*an2,64,sh,sw]
        disp_res = torch.cat([disp_init_fea, hr_fea], 1)
        # print('res ', disp_res.shape)
        disp_res = self.refine_res(disp_res).view(N, an2, sh, sw)  # [N*an2,1,sh,se]->[N,an2,sh,se]

        out = disp_init + disp_res

        return out, mask
    

class WarpNet(nn.Module):
    
    def __init__(self, opt):
        
        super(WarpNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.disp_net = DepthNet(an=opt.angular_num, scale=opt.scale)

    def forward(self, lr, hr):


        N, an2,  h, w = lr.shape  # lr [N,9,h,w]
        __, _, hs, ws = hr.shape # hr [N,1,hs,ws]
        an = int(math.sqrt(an2))
             
       # disparity estimation
        disp, mask = self.disp_net(lr, hr)

        # inverse warping
        warpedHR = torch.zeros(N, an2, hs, ws).type_as(lr)  #[N,81,hs,ws]
        for i in range(an2):
            warpedHR[:, i:i+1, :, :] = warp(disp[:, i, :, :], hr, i, an)

        return disp, warpedHR, mask
 


 