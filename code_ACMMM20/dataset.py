import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
import cv2
from scipy import misc
from math import ceil

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, scale, patch_size, an):
        super(DatasetFromHdf5, self).__init__()
        
        hf = h5py.File(file_path)
        self.img_label = hf.get('img_label') # [N,ah,aw,h,w]
        self.img_LR_2 = hf.get('img_LR_2')   # [N,ah,aw,h/2,w/2]
        self.img_LR_4 = hf.get('img_LR_4')   # [N,ah,aw,h/4,w/4]
        self.img_LR_8 = hf.get('img_LR_8')   # [N,ah,aw,h/8,w/8]
        # self.img_HR = hf.get('img_HR')       # [N,1,1,h,w]
        # self.img_HR_2 = hf.get('img_HR_2')   # [N,1,1,h/2,w/2]
        # self.img_HR_4 = hf.get('img_HR_4')   # [N,1,1,h/2,w/2]        
        
        self.scale = scale        
        self.psize = patch_size
        self.an = an
    
    def __getitem__(self, index):
                        
        # get one item
        label = self.img_label[index] # [ah,aw,h,w]
        lr_2 = self.img_LR_2[index]   # [ah,aw,h/2,w/2]
        lr_4 = self.img_LR_4[index]   # [ah,aw,h/4,w/4]
        lr_8 = self.img_LR_8[index]   # [ah,aw,h/8,w/8] 

        # crop to patch
        H = label.shape[2]
        W = label.shape[3]

        x = random.randrange(0, H-self.psize, 8)    
        y = random.randrange(0, W-self.psize, 8) 
        label = label[:, :, x:x+self.psize, y:y+self.psize] # [ah,aw,ph,pw]
        lr_2 = lr_2[:, :, x//2:x//2+self.psize//2, y//2:y//2+self.psize//2] # [ah,aw,ph/2,pw/2]    
        lr_4 = lr_4[:, :, x//4:x//4+self.psize//4, y//4:y//4+self.psize//4] # [ah,aw,ph/4,pw/4]  
        lr_8 = lr_8[:, :, x//8:x//8+self.psize//8, y//8:y//8+self.psize//8] # [ah,aw,ph/8,pw/8]  

        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            label = np.flip(np.flip(label,0),2)
            lr_2 = np.flip(np.flip(lr_2,0),2)
            lr_4 = np.flip(np.flip(lr_4,0),2)  
            lr_8 = np.flip(np.flip(lr_8,0),2)
            
        if np.random.rand(1)>0.5:
            label = np.flip(np.flip(label,1),3)
            lr_2 = np.flip(np.flip(lr_2,1),3)
            lr_4 = np.flip(np.flip(lr_4,1),3) 
            lr_8 = np.flip(np.flip(lr_8,1),3)
            
        # rotate
        r_ang = np.random.randint(1,5)
        label = np.rot90(label,r_ang,(2,3))
        label = np.rot90(label,r_ang,(0,1))
        lr_2 = np.rot90(lr_2,r_ang,(2,3))
        lr_2 = np.rot90(lr_2,r_ang,(0,1))           
        lr_4 = np.rot90(lr_4,r_ang,(2,3))
        lr_4 = np.rot90(lr_4,r_ang,(0,1)) 
        lr_8 = np.rot90(lr_8,r_ang,(2,3))
        lr_8 = np.rot90(lr_8,r_ang,(0,1))

        # get HR RGB
        ind  = ceil(self.an/2.)-1
        hr   = label[ind,ind,:,:]
        hr_2 = lr_2[ind,ind,:,:]
        hr_4 = lr_4[ind,ind,:,:]

        # to tensor     
        label = label.reshape(-1,self.psize,self.psize)     # [an,ph,pw]
        lr_2 = lr_2.reshape(-1,self.psize//2,self.psize//2) #[an,phs,pws]
        lr_4 = lr_4.reshape(-1,self.psize//4,self.psize//4) # [an,phs,pws]
        lr_8 = lr_8.reshape(-1,self.psize//8,self.psize//8) # [an,ph,pw]
        hr   = hr.reshape(-1,self.psize,self.psize)
        hr_2 = hr_2.reshape(-1,self.psize//2,self.psize//2)
        hr_4 = hr_4.reshape(-1,self.psize//4,self.psize//4)

        label = torch.from_numpy(label.astype(np.float32)/255.0)
        lr_2 = torch.from_numpy(lr_2.astype(np.float32)/255.0)  
        lr_4 = torch.from_numpy(lr_4.astype(np.float32)/255.0)  
        lr_8 = torch.from_numpy(lr_8.astype(np.float32)/255.0)
        hr   = torch.from_numpy(hr.astype(np.float32)/255.0)
        hr_2 = torch.from_numpy(hr_2.astype(np.float32)/255.0)
        hr_4 = torch.from_numpy(hr_4.astype(np.float32)/255.0)
        return label,lr_2,lr_4,lr_8,hr,hr_2,hr_4

            
    def __len__(self):
        return self.img_label.shape[0]