
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np

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
    

class AltFilter(nn.Module):
    def __init__(self, an):
        super(AltFilter, self).__init__()
        
        self.an = an
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.angconv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)

    def forward(self,x):

        N,c,h,w = x.shape #[N*81,c,h,w]
        N = N // (self.an*self.an)
        
        out = self.relu(self.spaconv(x)) #[N*81,c,h,w]
        
        out = out.view(N,self.an*self.an,c,h*w)
        out = torch.transpose(out,1,3)
        out = out.view(N*h*w,c,self.an,self.an)  #[N*h*w,c,9,9]

        out = self.relu(self.angconv(out)) #[N*h*w,c,9,9]
    
        out = out.view(N,h*w,c,self.an*self.an)
        out = torch.transpose(out,1,3)
        out = out.view(N*self.an*self.an,c,h,w) #[N*81,c,h,w]

        return out

        
class Conv_Relu_Block(nn.Module):

    def __init__(self,fn):
    
        super(Conv_Relu_Block, self).__init__()        
        self.conv = nn.Conv2d(in_channels = fn, out_channels = fn, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv(x))        
        return out
        
        
def make_Altlayer(layer_num, an):
    layers = []
    for i in range( layer_num ):
        layers.append( AltFilter(an) )
    return nn.Sequential(*layers)     


def make_Convlayer(layer_num, fn):
    layers = []
    for i in range( layer_num ):
        layers.append( Conv_Relu_Block(fn))            
    return nn.Sequential(*layers)       
        

class copyNet_4x(nn.Module):
    
    def __init__(self, an, layer):        
        
        super(copyNet_4x, self).__init__()
        
        self.an = an 
        self.an2 = an * an
        self.relu = nn.ReLU(inplace=True)
        
        # LR
        self.conv0 = nn.Conv2d(in_channels = 1,out_channels = 64,kernel_size=3,stride=1,padding=1)
        
        self.altblock1 = make_Altlayer(layer_num=layer, an=self.an)
        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res1 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup1 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)
        
        self.altblock2 = make_Altlayer(layer_num=layer, an=self.an)
        self.fup2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res2 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup2 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)       
        
        # HR
        self.conv0_hr1 = nn.Conv2d(in_channels=1, out_channels=self.an2, kernel_size=3, stride=1, padding=1)
        self.convblock1 = make_Convlayer(layer_num = layer, fn=self.an2)
        self.conv0_hr2 = nn.Conv2d(in_channels=1, out_channels=self.an2, kernel_size=3, stride=1, padding=1)
        self.convblock2 = make_Convlayer(layer_num=layer,fn=self.an2)
                           
        # fusion
        self.conv0_fusion1 = nn.Conv2d(in_channels = 2,out_channels = 64,kernel_size=3,stride=1,padding=1)
        #self.conv_fusion1 = make_Altlayer(layer_num = layer, an=self.an)
        self.conv_fusion1 = make_Convlayer(layer_num = layer, fn=64)
        self.res_fusion1 = nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size=3,stride=1,padding=1)

        self.conv0_fusion2 = nn.Conv2d(in_channels = 2,out_channels = 64,kernel_size=3,stride=1,padding=1)
        #self.conv_fusion2 = make_Altlayer(layer_num = layer, an=self.an)
        self.conv_fusion2 = make_Convlayer(layer_num = layer, fn=64)
        self.res_fusion2 = nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size=3,stride=1,padding=1)        
        
        self.conv_mask = nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size=3,stride=1,padding=1)
        
        for m in self.modules():
            # print(m)
            # if isinstance(m, nn.Conv2d):
                # print('conv')
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data_normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                    # m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                # print('Transconv')
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
        
        
        
    def forward(self, lr, hr, hr_2):
        # hr [N,1,4h,4w]
        N,_,h,w = lr.shape   #lr [N,81,h,w]  
        lr = lr.view(N*self.an2,1,h,w)  #[N*81,1,h,w]                          
       
        # hr feature        
        hr_feat_4x = self.relu(self.conv0_hr1(hr)) #[N,81,4h,4w]
        hr_feat_4x = self.convblock1(hr_feat_4x) #[N,81,4h,4w]
        
        hr_feat_2x = self.relu(self.conv0_hr2(hr_2)) #[N,81,2h,2w]
        hr_feat_2x = self.convblock2(hr_feat_2x) #[N,81,2h,2w]        
        
        hr_feat_4x = hr_feat_4x.view(N*self.an2,1,h*4,w*4)
        hr_feat_2x = hr_feat_2x.view(N*self.an2,1,h*2,w*2)       
       
        # x2
        x = self.relu(self.conv0(lr)) #[N*81,64,h,w]
        f_1 = self.altblock1(x) #[N*81,64,h,w]
        fup_1 = self.fup1(f_1)  #[N*81,64,2h,2w]
        lr_feat_1 = self.relu(self.res1(fup_1)) #[N*81,1,2h,2w]
        
        fcat_1 = torch.cat((lr_feat_1,hr_feat_2x),1) #[N*81,2,2h,2w]
        fcat_1 = self.relu(self.conv0_fusion1(fcat_1)) #[N*81,64,2h,2w]
        fcat_1 = self.conv_fusion1(fcat_1)  #[N*81,64,2h,2w]
        res_1 = self.res_fusion1(fcat_1) #[N*81,1,2h,2w]
             
        iup_1 = self.iup1(lr)   #[N*81,1,2h,2w]
        sr_2x = res_1 + iup_1   #[N*81,1,2h,2w]
         
        #x4 
        f_2 = self.altblock2(fup_1)  #[N*81,64,2h,2w]
        fup_2 = self.fup2(f_2)     #[N*81,64,4h,4w]
        lr_feat_2 = self.relu(self.res2(fup_2)) #[N*81,1,4h,4w]
        
        fcat_2 = torch.cat((lr_feat_2,hr_feat_4x),1) #[N*81,2,4h,4w]
        fcat_2 = self.relu(self.conv0_fusion2(fcat_2)) #[N*81,64,4h,4w]
        fcat_2 = self.conv_fusion2(fcat_2) 
        res_2 = self.res_fusion2(fcat_2) #[N*81,1,4h,4w]

        iup_2 = self.iup2(sr_2x)  #[N*81,1,4h,4w]
        
        sr_4x = res_2 + iup_2   #[N*81,1,4h,4w]

        
        sr_2x = sr_2x.view(N,self.an2,h*2,w*2)
        sr_4x = sr_4x.view(N,self.an2,h*4,w*4)   
        
        mask = self.conv_mask(fcat_2)
        mask = mask.view(N,self.an2,h*4,w*4)
        
        return sr_2x,sr_4x,mask       
        
    
    
class copyNet_8x(nn.Module):
    
    def __init__(self, an, layer):        
        
        super(copyNet_8x, self).__init__()
        
        self.an = an 
        self.an2 = an * an
        self.relu = nn.ReLU(inplace=True)
        
        #LR
        self.conv0 = nn.Conv2d(in_channels = 1,out_channels = 64,kernel_size=3,stride=1,padding=1)
        
        self.altblock1 = make_Altlayer(layer_num=layer, an=self.an)
        self.fup1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res1 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup1 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)
        
        self.altblock2 = make_Altlayer(layer_num=layer,an=self.an)
        self.fup2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res2 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup2 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)
        
        self.altblock3 = make_Altlayer(layer_num=layer,an=self.an)
        self.fup3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64,out_channels = 64,kernel_size=4,stride=2,padding=1),
            nn.ReLU(inplace=True),
            )
        self.res3 =  nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)    
        self.iup3 =  nn.ConvTranspose2d(in_channels = 1,out_channels = 1,kernel_size=4,stride=2,padding=1)

        #HR        
        self.conv0_hr1 = nn.Conv2d(in_channels=1, out_channels=self.an2, kernel_size=3, stride=1, padding=1)
        self.convblock1 = make_Convlayer(layer_num = layer, fn=self.an2)
        self.conv0_hr2 = nn.Conv2d(in_channels=1, out_channels=self.an2, kernel_size=3, stride=1, padding=1)
        self.convblock2 = make_Convlayer(layer_num=layer,fn=self.an2)
        self.conv0_hr3 = nn.Conv2d(in_channels=1, out_channels=self.an2, kernel_size=3, stride=1, padding=1)
        self.convblock3 = make_Convlayer(layer_num=layer,fn=self.an2)          

        # fusion
        self.conv0_fusion1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_fusion1 = make_Convlayer(layer_num=layer, fn=64)
        self.res_fusion1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.conv0_fusion2 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_fusion2 = make_Convlayer(layer_num=layer, fn=64)
        self.res_fusion2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.conv0_fusion3 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_fusion3 = make_Convlayer(layer_num=layer, fn=64)
        self.res_fusion3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        self.conv_mask = nn.Conv2d(in_channels = 64,out_channels = 1,kernel_size=3,stride=1,padding=1)
        
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data_normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                    # m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()        
        
        
        
    def forward(self, lr, hr, hr_2, hr_4):
        # hr [N,1,8h,8w]
        N,_,h,w = lr.shape   #lr [N,81,h,w]  
        lr = lr.view(N*self.an2,1,h,w)  #[N*81,1,h,w]                          

        # hr feature        
        hr_feat_8x = self.relu(self.conv0_hr1(hr)) #[N,81,8h,8w]
        hr_feat_8x = self.convblock1(hr_feat_8x)  #[N,81,8h,8w]
        
        hr_feat_4x = self.relu(self.conv0_hr2(hr_2)) #[N,81,4h,4w]
        hr_feat_4x = self.convblock2(hr_feat_4x) #[N,81,4h,4w]
        
        hr_feat_2x = self.relu(self.conv0_hr3(hr_4)) #[N,81,2h,2w]
        hr_feat_2x = self.convblock3(hr_feat_2x) #[N,81,2h,2w]
        
        hr_feat_8x = hr_feat_8x.view(N*self.an2,1,h*8,w*8)
        hr_feat_4x = hr_feat_4x.view(N*self.an2,1,h*4,w*4)
        hr_feat_2x = hr_feat_2x.view(N*self.an2,1,h*2,w*2)       
                      
        
        #x2 
        x = self.relu(self.conv0(lr)) #[N*81,64,h,w]       
        f_1 = self.altblock1(x) #[N*81,64,h,w]
        fup_1 = self.fup1(f_1)  #[N*81,64,2h,2w]
        lr_feat_1 = self.relu(self.res1(fup_1)) #[N*81,1,2h,2w]
        
        fcat_1 = torch.cat((lr_feat_1,hr_feat_2x),1) #[N*81,2,2h,2w]  
        fcat_1 = self.relu(self.conv0_fusion1(fcat_1)) #[N*81,64,2h,2w]
        fcat_1 = self.conv_fusion1(fcat_1)
        res_1 = self.res_fusion1(fcat_1) #[N*81,1,2h,2w]
        
        iup_1 = self.iup1(lr)   #[N*81,1,2h,2w]
        sr_2x = res_1 + iup_1   #[N*81,1,2h,2w]
        
        #x4  
        f_2 = self.altblock2(fup_1)  #[N*81,64,2h,2w]
        fup_2 = self.fup2(f_2)     #[N*81,64,4h,4w]
        lr_feat_2 = self.relu(self.res2(fup_2)) #[N*81,1,4h,4w]
        
        fcat_2 = torch.cat((lr_feat_2,hr_feat_4x),1)   #[N*81,2,4h,4w]
        fcat_2 = self.relu(self.conv0_fusion2(fcat_2)) #[N*81,64,4h,4w]
        fcat_2 = self.conv_fusion2(fcat_2) 
        res_2 = self.res_fusion2(fcat_2) #[N*81,1,4h,4w]
        
        iup_2 = self.iup2(sr_2x)  #[N*81,1,4h,4w]
        sr_4x = res_2 + iup_2   #[N*81,1,4h,4w]
        
        
        #x8
        f_3 = self.altblock3(fup_2)
        fup_3 = self.fup3(f_3)
        lr_feat_3 = self.relu(self.res3(fup_3)) 
        
        fcat_3 = torch.cat((lr_feat_3,hr_feat_8x),1)
        fcat_3 = self.relu(self.conv0_fusion3(fcat_3))
        fcat_3 = self.conv_fusion3(fcat_3)
        res_3 = self.res_fusion3(fcat_3)
        
        iup_3 = self.iup3(sr_4x)
        sr_8x = res_3 + iup_3  #[N*81,1,8h,8w]
        
        sr_2x = sr_2x.view(N,self.an2,h*2,w*2)
        sr_4x = sr_4x.view(N,self.an2,h*4,w*4)   
        sr_8x = sr_8x.view(N,self.an2,h*8,w*8)
        
        mask = self.conv_mask(fcat_3)
        mask = mask.view(N,self.an2,h*8,w*8)
        
        return sr_2x,sr_4x,sr_8x, mask
 


 