
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import math

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1 
    else:
        center = factor - 0.5
        
    og = np.ogrid[:size, :size]
    filter = ( 1 - abs(og[0] - center) / factor ) * \
             ( 1 - abs(og[1] - center) / factor )
             
    return torch.from_numpy(filter).float()
    

class Conv_Relu_Block(nn.Module):
    def __init__(self, fn):
        super(Conv_Relu_Block, self).__init__()        
        self.conv = nn.Conv2d(in_channels = fn, out_channels = fn, kernel_size = 3, stride = 1, padding = 1)        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv(x))        
        return out

def make_Convlayer(layer_num, fn):
    layers = []
    for i in range( layer_num ):
        layers.append( Conv_Relu_Block(fn))            
    return nn.Sequential(*layers)       
        

class Dense_Block(nn.Module):
    def __init__(self, in_ch, grow_ch, out_ch, layer_num):
        super(Dense_Block, self).__init__()

        self.bottlenect = nn.ModuleList([
            nn.Conv2d(in_ch+i*grow_ch, grow_ch, 1, 1, 0, bias=True) for i in range(layer_num)
        ])
        self.transm = nn.Conv2d(in_ch+layer_num*grow_ch, out_ch, 1, 1, 0, bias=True)

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
        return out


class SRNet(nn.Module):
    
    def __init__(self, opt):
        
        super(SRNet, self).__init__()
        self.scale = opt.scale
        
        self.an = opt.angular_num
        self.an2 = opt.angular_num  * opt.angular_num
        self.relu = nn.ReLU(inplace=True)
        
        # LR
        self.lr_dense = Dense_Block(in_ch=self.an2-1, grow_ch=64, out_ch=64*self.an2, layer_num=12)

        up_mask = []
        for _ in range(int(math.log(opt.scale, 2))):
            up_mask.append(nn.Conv2d(64, 4 * 64, 3, 1, 1, bias=True))
            up_mask.append(nn.PixelShuffle(2))
            up_mask.append(nn.ReLU(inplace=True))
        self.lr_up = nn.Sequential(*up_mask)

        # HR
        self.hr_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.hr_convblock = make_Convlayer(layer_num=6, fn=64)

        # fusion
        self.fusion_dense = Dense_Block(in_ch=128, grow_ch=64, out_ch=64, layer_num=6)
        self.fusion_res = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.conv_mask = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)


    def forward(self, lr, hr):
        # hr [N,1,sh,sw]
        # lr [N,9,h,w]

        N, an2, h, w = lr.shape
        an = int(math.sqrt(an2))
        ind_cv = torch.tensor(an // 2 * an + an // 2)
        lr_side = torch.cat([lr[:, :ind_cv], lr[:, ind_cv+1:]], 1).view(N, an2-1, h, w)  # [N,an2-1,h,w]


        # hr feature        
        hr_feat = self.relu(self.hr_conv(hr.view(N, 1, self.scale*h, self.scale*w)))  # [N,64,sh,sw]
        hr_feat = self.hr_convblock(hr_feat)  # [N,64,sh,sw]
        hr_feat = hr_feat.repeat(self.an2, 1, 1, 1) #[N*an2,64,sh,se]

        # lr
        # x = self.lr_dense(lr_side.view(N, 3*(an2-1), h, w))  # [N,an2*64,h,w]
        x = self.relu(self.lr_dense(lr_side))
        lr_feat = self.lr_up(x.view(N*an2, 64, h, w))  # [N*an2,64,sh,sw]

        # fusion
        fuse_feat = torch.cat((lr_feat, hr_feat), 1)  # [N*an2,128,sh,sw]
        fuse_feat = self.fusion_dense(fuse_feat)  #[N*9,64,sh,sw]
        fuse_res = self.fusion_res(self.relu(fuse_feat)) #[N*9,1,sh,sw]


        # res
        iup = functional.interpolate(lr.view(N*an2, 1, h, w), scale_factor=self.scale, mode='bilinear',align_corners=False)
        sr = fuse_res + iup   # [N*an2,3,2h,2w]
        sr = sr.view(N, an2, h*self.scale, w*self.scale)

        mask = self.conv_mask(fuse_feat).view(N, an2, h*self.scale, w*self.scale)

        return sr, mask



    

