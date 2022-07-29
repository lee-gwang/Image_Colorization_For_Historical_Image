# lib
import numpy as np
import pandas as pd
import random
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd
import glob
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
parser.add_argument('--gpu', '-gpu', type=str, default='2,3,4,5', help='gpu')

# related models
parser.add_argument('--batch_size', '-bs', type=int, default=256)
parser.add_argument('--backbone', type=str, default='efficientnet-b0', help='')
parser.add_argument('--decoder', type=str, default='Unet', help='')

# else
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--wandb', '-wb', action='store_true', help='use wandb')
parser.add_argument('--exp_comment', '-expc', type=str, default='version0', help='experiments folder comment')
parser.add_argument('--saved_models', '-sm',type=str, default='')

# path
parser.add_argument('--img_path', '-ip',type=str, default='')
parser.add_argument('--depth', type=int, default=1)

args = parser.parse_args()


class CFG:
    amp           = True
    seed          = 101
    debug         = args.debug # set debug=False for Full Training
    wandb         = args.wandb
    model_name    = ['Unet'] # decoder
    backbone      = [ args.backbone] # encoder # LeViT_UNet_384 #efficientnet-b2 # 'se_resnext50_32x4d'
    add_comment   = f'{args.exp_comment}'#'negative-5k-bs32'

    #comment       = f'{model_name}-{backbone}-320x384'
    num_channel   = 1
    num_classes   = 2

    train_bs      = args.batch_size
    valid_bs      = train_bs
    img_size      = [224, 224]#[320, 384]
    #
    n_fold        = 5
    folds         = [0]
    gpu           = args.gpu
    saved_models  = args.saved_models
    img_path      = args.img_path
    path_depth    = args.depth

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
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    def unnorm(norm_img):
        norm_img[:,0,:,:]  = norm_img[:,0,:,:] * 100. + 50.
        norm_img[:,1:,:,:] = norm_img[:,1:,:,:]*110.
        return norm_img

    y_preds=[]
    for step, gray_imgs in pbar:        
        gray_imgs  = gray_imgs.to(device)#, dtype=torch.float)
        
        batch_size = gray_imgs.size(0)
        
        y_pred  = model(gray_imgs)
        y_pred = torch.cat([gray_imgs,y_pred], 1)
        
        y_pred = y_pred.cpu().detach()*255 ; y_preds.append(y_pred)
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

    y_preds = torch.cat(y_preds, 0)
    
    return y_preds # 정확하지 않음, batch 남는 부분에 의해 / # 저장위해 y_pred


# ------------------------
#  Run
# ------------------------
def run_inference(model, df, run, device):
    # dataloader
    valid_df = df

    # new_aug 
    valid_dataset = ActivityDataset(valid_df, type='infer', label=False)

    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not CFG.debug else 20, 
                              num_workers=CFG.num_workers, shuffle=False, pin_memory=True)
    
    
    # To automatically log gradients
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
        
    y_pred = valid_one_epoch(model, valid_loader, 
                                                device=CFG.device, 
                                                )
    
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
    for n, path_ in enumerate(valid_df['new_rgb_paths']):
        show_img = cv2.imread(path_)
        #show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        show_img = cv2.resize(show_img, (768,768))
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
    df  = pd.DataFrame()
    if CFG.path_depth == 2:
        path_ = glob.glob(os.path.join(CFG.img_path, '*/*'))
    elif CFG.path_depth == 1:
        path_ = glob.glob(os.path.join(CFG.img_path, '*'))
    df['new_rgb_paths'] = path_

    df = df.sample(1000)

    for encoder in CFG.backbone:
        for decoder in CFG.model_name:
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
            run_inference(model, df, run, CFG.device)
            
    
    return

if __name__ == '__main__':
    main(CFG)
