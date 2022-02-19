
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
from os import path
from os.path import join
from collections import defaultdict
from math import log10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import DatasetFromHdf5
from hybridSR_fusion import fusionNet_4x, fusionNet_8x
#--------------------------------------------------------------------------#
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR")

#training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default = 64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default = 1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--num_cp", type=int, default=5, help="Number of epoches for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epoches for saving loss figure")

#dataset
parser.add_argument("--dataset", type=str, default="", help="Dataset for training")
parser.add_argument("--angular_num", type=int, default=9, help="Size of angular dim")

#model 
parser.add_argument("--scale", type=int, default=8, help="SR factor")
parser.add_argument("--copy_layer", type=int, default=6, help="layer number for copyNet")
parser.add_argument("--warp_layer", type=int, default=3, help="layer number for warpNet")

opt = parser.parse_args()
print(opt)
#--------------------------------------------------------------------------#
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#--------------------------------------------------------------------------#
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

model_dir = 'model_x{}_{}_lr{}_step{}x{}_cL{}_wL{}'.format(opt.scale,opt.dataset,opt.lr,opt.step,opt.reduce,opt.copy_layer,opt.warp_layer)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
an = opt.angular_num
#--------------------------------------------------------------------------#
# Data loader
print('===> Loading datasets')
dataset_path = join('LFData', 'train_{}.h5'.format(opt.dataset))
train_set = DatasetFromHdf5(dataset_path, opt.scale, opt.patch_size, an)
train_loader = DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=True)
print('loaded {} LFIs from {}'.format(len(train_loader),dataset_path))
#--------------------------------------------------------------------------#
# Build model
print("building fusion net")
if opt.scale == 4 :
    model = fusionNet_4x(an,opt.scale,opt.copy_layer,opt.warp_layer).to(device)
elif opt.scale == 8:
    model = fusionNet_8x(an,opt.scale,opt.copy_layer,opt.warp_layer).to(device)
#-------------------------------------------------------------------------#
# optimizer and loss logger
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
losslogger = defaultdict(list)
#------------------------------------------------------------------------#    
# optionally resume from a checkpoint
if opt.resume_epoch:
    resume_path = join(model_dir,'model_epoch_{}.pth'.format(opt.resume_epoch))
    if os.path.isfile(resume_path):
        print("==>loading checkpoint 'epoch{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losslogger = checkpoint['losslogger']
    else:
        print("==> no model found at 'epoch{}'".format(opt.resume_epoch))
#------------------------------------------------------------------------#
# loss
def local_confidence_loss(sr, mask, target):    
    N,an,h,w = mask.shape     
    l2norm = mask.view(N,an,-1).norm(p=2,dim=2)
    mask_norm = mask / l2norm.view(N,an,1,1) 
    loss = torch.sum((sr - target)**2 * mask_norm) / torch.numel(sr)   
    # print(mask_norm.view(N,an,-1).norm(p=2,dim=2))
    # print(sr.shape)
    # print(mask_norm.shape)
    return loss

def reconstruction_loss(X,Y):
# L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss
#-----------------------------------------------------------------------#  
 
def train_4x(epoch): 
    model.train()
    scheduler.step()    
    loss_count = 0.   
    for k in range(50):  
        for i,batch in enumerate(train_loader,1):
            #print(i)
            label,lr_2,lr_4,_,hr,hr_2,_  = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device), batch[4].to(device),batch[5].to(device),batch[6].to(device)
            
            # forward pass 
            sr_copy_2x, sr_copy_4x, mask_copy, _, sr_warp, mask_warp, sr_fusion = model(lr_4, hr, hr_2)
            
            # copy loss
            loss_copy = reconstruction_loss(sr_copy_2x, lr_2) + reconstruction_loss(sr_copy_4x, label) + local_confidence_loss(sr_copy_4x.detach(), mask_copy, label)
            loss_warp = reconstruction_loss(sr_warp, label) + local_confidence_loss(sr_warp.detach(), mask_warp, label)
            loss_fusion = reconstruction_loss(sr_fusion, label)            
            loss = loss_copy + loss_warp + loss_fusion

            loss_count += loss.item()   
            
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count/len(train_loader))       

def train_8x(epoch): 
    model.train()
    scheduler.step()    
    loss_count = 0.   
    for k in range(50):  
        for i,batch in enumerate(train_loader,1):
            #print(i)
            label,lr_2,lr_4,lr_8,hr,hr_2,hr_4  = batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device), batch[4].to(device),batch[5].to(device),batch[6].to(device)
            
            # forward pass 
            sr_copy_2x, sr_copy_4x, sr_copy_8x, mask_copy, _, sr_warp, mask_warp, sr_fusion = model(lr_8, hr, hr_2, hr_4)
            
            # copy loss
            loss_copy = reconstruction_loss(sr_copy_2x, lr_4) + reconstruction_loss(sr_copy_4x, lr_2) + reconstruction_loss(sr_copy_8x, label) + local_confidence_loss(sr_copy_8x.detach(), mask_copy, label)
            loss_warp = reconstruction_loss(sr_warp, label) + local_confidence_loss(sr_warp.detach(), mask_warp, label)
            loss_fusion = reconstruction_loss(sr_fusion, label)            
            loss = loss_copy + loss_warp + loss_fusion

            loss_count += loss.item()   
            
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count/len(train_loader))     

#-------------------------------------------------------------------------#
print('==>training')
for epoch in range(opt.resume_epoch+1, 100000): 

    if opt.scale == 4:   
        train_4x(epoch)
    elif opt.scale == 8:
        train_8x(epoch)
        
#     checkpoint
    if epoch % opt.num_cp == 0:        
        model_save_path = join(model_dir,"model_epoch_{}.pth".format(epoch))        
        state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'losslogger': losslogger,}
        torch.save(state,model_save_path)
        print("checkpoint saved to {}".format(model_save_path))     

    if epoch % opt.num_snapshot == 0:   
        plt.figure()
        plt.title('loss')
        plt.plot(losslogger['epoch'],losslogger['loss'])
        plt.savefig(model_dir+".jpg")
        plt.close()
        

