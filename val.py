# lib
import numpy as np
import pandas as pd
import random
from glob import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

#import rasterio
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

import wandb
import segmentation_models_pytorch as smp
import math
from dataloader import ImageNetDataset, ActivityDataset
from utils.utils import load_data, init_logger

# model
from models.seg_p import build_model

import argparse

parser = argparse.ArgumentParser(description='Script of SKT Colorization')
parser.add_argument('--dataset', '-dt', type=str, default='activitynet_384', help='activitynet/imagenet20k')
parser.add_argument('--gpu', '-gpu', type=str, default='2,3,4,5', help='gpu')

parser.add_argument('--batch_size', '-bs', type=int, default=256)
parser.add_argument('--epochs', '-e', type=int, default=50)

# related models
parser.add_argument('--backbone', type=str, default='efficientnet-b0', help='')
parser.add_argument('--decoder', type=str, default='Unet', help='')

# related optimizer & scheduler ..
parser.add_argument('--optimizer', type=str, default='adam', help='')
parser.add_argument('--loss', type=str, default='mae', help='mae/mse')
parser.add_argument('--scheduler', type=str, default='CosineAnnealingWarmRestarts', help='CosineAnnealingWarmRestarts/CosineAnnealingLR')


# else
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--wandb', '-wb', action='store_true', help='use wandb')
parser.add_argument('--exp_comment', '-expc', type=str, default='version0', help='experiments folder comment')
parser.add_argument('--save_img', action='store_true', help='save images')
parser.add_argument('--saved_models', '-sm',type=str, default='')

parser.add_argument('--test', action='store_true', help='test')



args = parser.parse_args()


class CFG:
    dataset       = args.dataset
    amp           = True
    test          = True
    seed          = 101
    debug         = args.debug # set debug=False for Full Training
    wandb         = args.wandb
    model_name    = ['Unet'] # decoder
    backbone      = [ args.backbone] # encoder # LeViT_UNet_384 #efficientnet-b2 # 'se_resnext50_32x4d'
    add_comment   = f'{dataset}-{args.exp_comment}'#'negative-5k-bs32'

    #comment       = f'{model_name}-{backbone}-320x384'
    num_channel   = 1
    num_classes   = 2

    train_bs      = args.batch_size
    valid_bs      = train_bs
    img_size      = [224, 224]#[320, 384]
    epochs        = args.epochs
    lr            = 2e-3
    scheduler     = args.scheduler #'CosineAnnealingWarmRestarts'#'CosineAnnealingLR'
    optimizer     = args.optimizer
    loss          = args.loss
    #
    warmup_factor = 5 # warmupv2
    warmup_epo    = 2
    cosine_epo    = 13
    #
    min_lr        = 1e-6
    T_max         = epochs#int(30000/train_bs*epochs)+50
    T_0           = epochs//3
    warmup_epochs = 0
    wd            = 1e-6
    n_accumulate  = 1#max(1, 32//train_bs)
    n_fold        = 5
    folds         = [0]
    gpu           = args.gpu
    save_img      = args.save_img #True/False # save valid predict images
    saved_models  = args.saved_models
    num_workers   = 8
    #wb_key        = '370d7a23c0cf0998cc65127c6a7bf00180540617'
    
def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

if CFG.scheduler=='warmupv2':
    assert CFG.epochs == (CFG.warmup_epo + CFG.cosine_epo)
# seed
set_seed(CFG.seed)

# use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = CFG.gpu
CFG.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------
#  Loss & Metric
# ------------------------
def rmse_score(true, pred):
    score = math.sqrt(np.mean((true-pred)**2))
    return score

def psnr_score(true, pred, pixel_max):
    score = 20*np.log10(pixel_max/rmse_score(true, pred))
    return score

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

def loss_fn(CFG):
    if CFG.loss == 'mae':
        loss = nn.L1Loss()
    elif CFG.loss == 'mse':
        loss = nn.MSELoss()
    return loss

# ------------------------
#  Val
# ------------------------
@torch.no_grad()
def valid_one_epoch(model, dataloader, device):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    psnr_score   = 0.
    
    val_scores = []
    criterion = loss_fn(CFG)
    psnr_metric = PSNR()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    def unnorm(norm_img):
        norm_img[:,0,:,:]  = norm_img[:,0,:,:] * 100. + 50.
        norm_img[:,1:,:,:] = norm_img[:,1:,:,:]*110.
        return norm_img

    y_preds=[]
    for step, (gray_imgs, rgb_imgs) in pbar:        
        gray_imgs  = gray_imgs.to(device)#, dtype=torch.float)
        rgb_imgs   = rgb_imgs.to(device)#, dtype=torch.float)
        
        batch_size = gray_imgs.size(0)
        
        y_pred  = model(gray_imgs)
        y_pred = torch.cat([gray_imgs,y_pred], 1)
        loss    = criterion(y_pred, rgb_imgs)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        

        # bs, c, h, w
        # normalize
        
        #y_pred = unnorm(y_pred.cpu().detach())
        y_pred = y_pred.cpu().detach()*255 ; y_preds.append(y_pred)
        rgb_imgs = rgb_imgs.cpu().detach()*255.
        psnr_score += psnr_metric(rgb_imgs, y_pred)
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        gpu_memory=f'{mem:0.2f} GB')

    y_preds = torch.cat(y_preds, 0)
    
    return epoch_loss, psnr_score/len(dataloader), y_preds # 정확하지 않음, batch 남는 부분에 의해 / # 저장위해 y_pred


# ------------------------
#  Run
# ------------------------
def run_inference(model, df, run, device, fold):
    # dataloader
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    if CFG.debug:
        valid_df = valid_df.head(32*3)
    # new_aug 
    if CFG.test:
        valid_dataset = ActivityDataset(valid_df, type='valid', label=False)
    else:
        valid_dataset = ActivityDataset(valid_df, type='valid')

        

    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not CFG.debug else 20, 
                              num_workers=CFG.num_workers, shuffle=False, pin_memory=True)
    
    
    # To automatically log gradients
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
        
    val_loss, val_psnr, y_pred = valid_one_epoch(model, valid_loader, 
                                                device=CFG.device, 
                                                )
    
    if CFG.save_img:
        # pred
        y_pred = y_pred.permute(0,2,3,1)
        y_pred = y_pred.cpu().detach().numpy()
        y_pred[y_pred>255.] = 255
        y_pred[y_pred<0] = 0
        y_pred = y_pred.astype('uint8')#*255

        # true
        os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'color'), exist_ok=True)
        os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'gray'), exist_ok=True)
        for n, show_img in enumerate(y_pred):
            show_img = cv2.cvtColor(show_img, cv2.COLOR_LAB2RGB)
            # show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(CFG.OUTPUT_DIR, f'pred/{n}.jpg'), show_img)
        for n, path_ in enumerate(valid_df['hr_rgb_paths']):
            show_img = cv2.imread(path_)
            show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
            # show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(CFG.OUTPUT_DIR, f'color/{n}.jpg'), show_img)
        for n, show_img in enumerate(y_pred):
            show_img = cv2.cvtColor(show_img, cv2.COLOR_LAB2RGB)
            show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(CFG.OUTPUT_DIR, f'gray/{n}.jpg'), show_img)

    return

if CFG.wandb:
    wandb.login(key=CFG.wb_key)

def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


# ------------------------
#  Main
# ------------------------
def main(CFG):

    """
    Prepare: 1.train 
    """
    if CFG.test:

    else:
        df = load_data(CFG)     
        df = df.groupby('video_id').sample(1) # video당 1개의 샘플만

    for encoder in CFG.backbone:
        for decoder in CFG.model_name:
            for fold in CFG.folds:
                print(f'#'*15)
                print(f'### Fold: {fold}')
                print(f'#'*15)
                CFG.comment = f'{decoder}-{encoder}-{CFG.add_comment}'

                # related of saved ########################################
                CFG.OUTPUT_DIR = f'./output_dirs/{CFG.comment}' 
                if not os.path.exists(CFG.OUTPUT_DIR):
                    os.makedirs(CFG.OUTPUT_DIR)

                ###########################################################
                run=None

                # model
                model = build_model(CFG, encoder=encoder, decoder=decoder)
                if len(CFG.gpu)>1:
                    model = nn.DataParallel(model)
                    model.to(CFG.device)
                else:
                    model.to(CFG.device)

                model.load_state_dict(torch.load(CFG.saved_models))
                # run inference
                run_inference(model, df, run, CFG.device, fold)
                
    
    return

if __name__ == '__main__':
    main(CFG)
