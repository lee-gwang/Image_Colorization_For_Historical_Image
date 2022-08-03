import os
import argparse
from unittest import result

parser = argparse.ArgumentParser(description='Script of SKT Colorization')
parser.add_argument('--gpu', '-gpu', type=str, default='0,1,2,3', help='gpu')

# related models
parser.add_argument('--batch_size', '-bs', type=int, default=1)
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

# inference
parser.add_argument('--patch', '-p', action='store_true', help='patch-wise inference')
parser.add_argument('--image', '-i', action='store_true', help='image-wise inference')



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

os.environ["CUDA_VISIBLE_DEVICES"] = CFG.gpu


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
from utils.utils import load_data, init_logger, set_seed

# model
from models.seg_p import build_model


# seed
set_seed(CFG.seed)

# use gpu
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
    results = []
    # m1 : 4가지 스케일 768_2, 768_4, 512, 1024 
    # m2 : 1가지 스케일 768_4
    # m3 : 오직 patch, 768_4
    # m4 : 오직 image-wise 768_4
    
    hp_list = [(4, 768, 740, 0.8, 0.2), (2, 768, 740, 0.8, 0.2), 
              (2, 1024, 1000, 0.7, 0.1), (3, 512, 500, 0.6, 0.1)] # (scale, image_size, stride, patch weight, image weight)



    # 그냥 튜플로 만들자 앙상블 파라미터
    batch_size = 128
    for step, gray_imgs in pbar:       
        ensemble_img = []
        len_weights = 0 # total of the ensemble weights
        if args.patch:
            for (scale, img_size, stride, p_w, i_w) in hp_list:
                img_size2 = img_size * scale
                _,c,h,w = gray_imgs.shape
                ##############
                crop = []
                position = []
                batch_count = 0

                result_img = np.zeros([3, h, w])
                voting_mask = np.zeros([3, h, w])
                for top in range(0, h, stride):
                    for left in range(0, w, stride):#-img_size+stride
                        piece = torch.zeros([1, 1, img_size2, img_size2])
                        temp = gray_imgs[:, :, top:top+img_size2, left:left+img_size2] # bs, c, h, w

                        piece[:, :, :temp.shape[2], :temp.shape[3]] = temp
                        # print('piece2 : ',piece.shape)

                        # crop.append(piece)
                        crop.append(piece)
                        position.append([top, left])
                        batch_count += 1
                        if batch_count == args.batch_size:
                            crop = torch.cat(crop, axis=0).to(device)
                            pred  = model(crop)
                            #
                            pred  = model(nn.Upsample(scale_factor=1/scale, mode='bilinear')(crop))
                            pred = nn.Upsample(scale_factor=scale, mode='bilinear')(pred)
                            #

                            pred = torch.cat([crop, pred], 1)
                            pred = pred.cpu().detach().numpy()
                            #pred = model(crop)*255
                            #pred = pred.detach().cpu().numpy()
                            crop = []
                            batch_count = 0
                            for num, (t, l) in enumerate(position):
                                piece = pred[num]
                                c_, h_, w_ = result_img[:, t:t+img_size2, l:l+img_size2].shape
                                result_img[:, t:t+img_size2, l:l+img_size2] += piece[:, :h_, :w_]
                                voting_mask[:, t:t+img_size2, l:l+img_size2] += 1
                            position = []
                if batch_count != 0: # batch size만큼 안채워지면
                    # crop = torch.from_numpy(np.array(crop)).permute(0,3,1,2).to(device)
                    crop = torch.stack(crop, axis=0).to(device)


                    # pred = model(crop)*255
                    # pred = pred.detach().cpu().numpy()
                    pred  = model(crop)
                    pred = torch.cat([crop,pred], 1)
                    pred = pred.cpu().detach().numpy()
                    crop = []
                    batch_count = 0
                    for num, (t, l) in enumerate(position):
                        piece = pred[num]
                        c_, h_, w_ = result_img[:, t:t+img_size2, l:l+img_size2].shape
                        result_img[:, t:t+h, l:l+w] += piece[:h, :w]
                        voting_mask[:, t:t+h, l:l+w] += 1
                    position = []


                result_img = result_img/voting_mask
                ensemble_img.append(result_img*p_w)
                len_weights += p_w


            if args.image:
                # image-wise ensmeble (나중에는, for구문 안에 넣어야할듯,,)
                gray_imgs = gray_imgs.to(device)
                pred = model(nn.Upsample(size=img_size, mode='bilinear')(gray_imgs))
                pred = nn.Upsample(size=(h,w), mode='bilinear')(pred)
                pred = torch.cat([gray_imgs, pred], 1).cpu().detach().numpy() # bs, c, h, w
                ensemble_img.append(pred[0]*i_w)
                len_weight += i_w



        # ensmeble
        # result_img = np.sum(ensemble_img, axis=0)/len(ensemble_img)
        result_img = np.sum(ensemble_img, axis=0)/len_weight

        result_img = np.around(result_img*255).astype(np.uint8).transpose(1,2,0)
        results.append(result_img)

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0

    
    return results # 정확하지 않음, batch 남는 부분에 의해 / # 저장위해 y_pred


# ------------------------
#  Run
# ------------------------
def run_inference(model, df, run, device):
    # dataloader
    valid_df = df

    # new_aug 
    valid_dataset = ActivityDataset(valid_df, type='patch_infer', label=False)

    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not CFG.debug else 20, 
                              num_workers=CFG.num_workers, shuffle=False, pin_memory=True)
    
    
    # To automatically log gradients
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
        
    y_pred = valid_one_epoch(model, valid_loader, 
                                                device=CFG.device, 
                                                )
    

    # true
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'color'), exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, 'gray'), exist_ok=True)
    for n, show_img in enumerate(y_pred):
        show_img = cv2.cvtColor(show_img, cv2.COLOR_LAB2RGB)
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

    # df = df.sample(1000)

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
