
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

from Dataset_Real import DatasetFromHdf5
from Model_Fusion import FusionNet
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
parser.add_argument("--dataset_path", type=str, default="", help="Dataset for training")
parser.add_argument("--angular_num", type=int, default=9, help="Size of angular dim")


parser.add_argument("--smooth", type=float, default=0.01, help="smooth loss weight")

#model 
parser.add_argument("--scale", type=int, default=2, help="SR factor")

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

model_dir = 'model_{}_x{}_sm{}_lr{}_step{}x{}'.format(opt.dataset, opt.scale, opt.smooth, opt.lr, opt.step, opt.reduce)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
an = opt.angular_num
#--------------------------------------------------------------------------#
# Data loader
print('===> Loading datasets')
# dataset_path = join('LFData', 'train_{}.h5'.format(opt.dataset))
train_set = DatasetFromHdf5(opt)
train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size,shuffle=True)
print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))
#--------------------------------------------------------------------------#
# Build model
print("building fusion net")
model = FusionNet(opt).to(device)
#-------------------------------------------------------------------------#
# optimizer and loss logger
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
losslogger = defaultdict(list)
#------------------------------------------------------------------------#    
# optionally resume from a checkpoint
if opt.resume_epoch:
    resume_path = join(model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
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
def reconstruction_loss(X,Y):
# L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss


def edge_aware_smoothness_loss(D, I, edge_constant=150):
    def img_grads(I):
        I_dy = I[:, :, 1:, :] - I[:, :, :-1, :]
        I_dx = I[:, :, :, 1:] - I[:, :, :, :-1]
        return I_dx, I_dy

    img_gx, img_gy = img_grads(I)
    weight_x = torch.exp(-edge_constant * torch.abs(img_gx))
    weight_y = torch.exp(-edge_constant * torch.abs(img_gy))
    # print('weight', weight_x.shape)
    # print('weight', weight_y.shape)

    disp_gx, disp_gy = img_grads(D)
    # print('disp gx', disp_gx.shape)
    # print('disp gy', disp_gy.shape)

    loss = (torch.mean(weight_x * torch.abs(disp_gx)) + torch.mean(weight_y * torch.abs(disp_gy)))/2.
    return loss

#-----------------------------------------------------------------------#  

def train(epoch):
    model.train()
    loss_count = 0.   
    for k in range(50):  
        for i, batch in enumerate(train_loader, 1):
            # print(i)
            label, lr_ct, hr = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            with torch.autograd.set_detect_anomaly(True):
                # forward pass
                sr_copy, mask_copy, disp, sr_warp, mask_warp, sr_fusion = model(lr_ct, hr)

                # copy loss
                loss_copy = reconstruction_loss(sr_copy, label)
                loss_warp = reconstruction_loss(sr_warp, label) + opt.smooth * edge_aware_smoothness_loss(disp, label)
                loss_fusion = reconstruction_loss(sr_fusion, label)
                loss = loss_copy + loss_warp + loss_fusion

                loss_count += loss.item()

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    scheduler.step()
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count/len(train_loader))     

#-------------------------------------------------------------------------#
print('==>training')
for epoch in range(opt.resume_epoch+1, 100000): 

    # if opt.scale == 4:
    #     train_4x(epoch)
    # elif opt.scale == 8:
    #     train_8x(epoch)

    train(epoch)
        
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
        

