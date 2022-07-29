# lib
import os

import argparse

parser = argparse.ArgumentParser(description='Script of SKT Colorization')
parser.add_argument('--dataset_path', '-dp', type=str, default='', help='train.csv path')
parser.add_argument('--gpu', '-gpu', type=str, default='2,3,4,5', help='gpu')

parser.add_argument('--batch_size', '-bs', type=int, default=256)
parser.add_argument('--epochs', '-e', type=int, default=50)
parser.add_argument('--lr', '-lr', type=float, default=2e-3, help='learning rate')


# related models
parser.add_argument('--backbone', type=str, default='efficientnet-b0', help='')
parser.add_argument('--decoder', type=str, default='Unet', help='')
parser.add_argument('--pretrained', type=str, default='None', help='')


# related optimizer & scheduler ..
parser.add_argument('--optimizer', type=str, default='adam', help='')
parser.add_argument('--loss', type=str, default='mae', help='mae/mse')
parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='CosineAnnealingWarmRestarts/CosineAnnealingLR')


# else
parser.add_argument('--val_iter',type=int, default=5)
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--wandb', '-wb', action='store_true', help='use wandb')
parser.add_argument('--exp_comment', '-expc', type=str, default='version0', help='experiments folder comment')
parser.add_argument('--save_img', action='store_true', help='save images')
# parser.add_argument('--show_num_img', type=int, default=50000, help='number of the show images')


args = parser.parse_args()


class CFG:
    dataset_path  = args.dataset_path
    amp           = True
    seed          = 101
    debug         = args.debug # set debug=False for Full Training
    wandb         = args.wandb
    model_name    = ['Unet'] # decoder
    backbone      = [ args.backbone] # encoder # LeViT_UNet_384 #efficientnet-b2 # 'se_resnext50_32x4d'
    pretrained    = args.pretrained # pretraianed models path
    add_comment   = f'{args.exp_comment}'#'negative-5k-bs32'

    #comment       = f'{model_name}-{backbone}-320x384'
    num_channel   = 1
    num_classes   = 2

    train_bs      = args.batch_size
    valid_bs      = train_bs
    img_size      = 768 # skt test size
    epochs        = args.epochs
    lr            = args.lr # 2e-3
    scheduler     = args.scheduler #'CosineAnnealingWarmRestarts'#'CosineAnnealingLR'
    optimizer     = args.optimizer
    loss          = args.loss
    val_iter      = args.val_iter
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
    #show_num_img  = args.show_num_img
    num_workers   = 16
    #wb_key        = '370d7a23c0cf0998cc65127c6a7bf00180540617'

os.environ["CUDA_VISIBLE_DEVICES"] = CFG.gpu

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

def save_img(y_pred, epoch, comment='val'):
    y_pred = y_pred.permute(0,2,3,1)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred[y_pred>255.] = 255
    y_pred[y_pred<0] = 0
    y_pred = y_pred.astype('uint8')#*255
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, comment, f'epoch{epoch}'),exist_ok=True)
    for n, show_img in enumerate(y_pred):
        show_img = cv2.cvtColor(show_img, cv2.COLOR_LAB2RGB)
        cv2.imwrite(os.path.join(CFG.OUTPUT_DIR, comment, f'epoch{epoch}/{n}.jpg'), show_img)


# ------------------------
#  Train
# ------------------------
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    criterion = loss_fn(CFG)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (gray_imgs, rgb_imgs) in pbar:         
        gray_imgs = gray_imgs.to(device)#, dtype=torch.float)
        rgb_imgs  = rgb_imgs.to(device)#, dtype=torch.float)
        
        batch_size = gray_imgs.size(0)

        # zero the parameter gradients
        optimizer.zero_grad()
        if CFG.amp:
            with amp.autocast(enabled=True):
                y_pred = model(gray_imgs)
                y_pred = torch.cat([gray_imgs,y_pred], 1)

                loss   = criterion(y_pred, rgb_imgs)
                loss   = loss / CFG.n_accumulate
            
            scaler.scale(loss).backward()
        
            if (step + 1) % CFG.n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            y_pred = model(gray_imgs)
            y_pred = torch.cat([gray_imgs,y_pred], 1)

            loss   = criterion(y_pred, rgb_imgs)
            loss   = loss / CFG.n_accumulate

            loss.backward()
            if (step + 1) % CFG.n_accumulate == 0:
                optimizer.step()

                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        #torch.cuda.empty_cache()
        #gc.collect()
    
    return epoch_loss

# ------------------------
#  Val
# ------------------------
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch, show=False):
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
        
        # y_pred = unnorm(y_pred.cpu().detach())
        # rgb_imgs = unnorm(rgb_imgs.cpu().detach())

        y_pred = y_pred.cpu().detach()*255 
        rgb_imgs = rgb_imgs.cpu().detach()*255.

        psnr_score += psnr_metric(rgb_imgs, y_pred)
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        gpu_memory=f'{mem:0.2f} GB')
        
        if show : y_preds.append(y_pred)
    if show: 
        y_preds = torch.cat(y_preds, 0)
        return y_preds
    
    return epoch_loss, psnr_score/len(dataloader) # 정확하지 않음, batch 남는 부분에 의해 / # 저장위해 y_pred

# ------------------------
#  Scheduler
# ------------------------
def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer,T_max=CFG.T_max, 
                                                   eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_0, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                    mode='min',
                                    factor=0.1,
                                    patience=7,
                                    threshold=0.0001,
                                    min_lr=CFG.min_lr,)
    elif CFG.scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        
    elif CFG.scheduler== 'warmupv2':
            scheduler_cosine=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, CFG.cosine_epo)
            scheduler_warmup=GradualWarmupSchedulerV2(optimizer, multiplier=CFG.warmup_factor, total_epoch=CFG.warmup_epo, after_scheduler=scheduler_cosine)
            scheduler=scheduler_warmup 
            
    elif CFG.scheduler == None:
        return None
        
    return scheduler


# ------------------------
#  Optimizer
# ------------------------
def get_optimizer(model, CFG):
    if CFG.optimizer =='adam':
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)

    elif CFG.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    else:
        print('Not support yet')
        return None
    return optimizer

# ------------------------
#  Run
# ------------------------
def run_training(model, df, skt_df, run, device, fold, LOGGER):
    # dataloader
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    show_df = df.query("fold==@fold").reset_index(drop=True).sample(500)



    if CFG.debug:
        train_df = train_df.head(200)
        valid_df = valid_df.head(50)
        show_df = valid_df.head(50)
        #skt_df = 

    # new_aug 

    train_dataset = ActivityDataset(train_df, type='train')
    valid_dataset = ActivityDataset(valid_df, type='valid')
    show_dataset = ActivityDataset(show_df, type='valid')
    show_dataset2 = ActivityDataset(skt_df, type='valid')


    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not CFG.debug else 20, 
                              num_workers=CFG.num_workers, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not CFG.debug else 20, 
                              num_workers=CFG.num_workers, shuffle=False, pin_memory=True)
    show_loader = DataLoader(show_dataset, batch_size=CFG.valid_bs if not CFG.debug else 20, 
                              num_workers=CFG.num_workers, shuffle=True, pin_memory=True)

    show_loader2 = DataLoader(show_dataset2, batch_size=CFG.valid_bs if not CFG.debug else 20, 
                              num_workers=CFG.num_workers, shuffle=True, pin_memory=True)
    # optimizer
    optimizer = get_optimizer(model, CFG)

    # scheduler
    scheduler = fetch_scheduler(optimizer)
    
    # To automatically log gradients
    if CFG.wandb:
        wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_psnr      = -np.inf
    best_epoch     = -1
    #history = defaultdict(list)
    
    for epoch in range(1, CFG.epochs + 1): 
        print(f'Epoch {epoch}/{CFG.epochs}', end='')
        train_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CFG.device, epoch=epoch)
        
        if epoch%CFG.val_iter ==0:
            val_loss, val_psnr = valid_one_epoch(model, valid_loader, 
                                                    device=CFG.device, 
                                                    epoch=epoch)
            y_pred = valid_one_epoch(model, show_loader, 
                                                    device=CFG.device, 
                                                    epoch=epoch, show=True)
            y_pred2 = valid_one_epoch(model, show_loader2, 
                                                    device=CFG.device, 
                                                    epoch=epoch, show=True)

            # Log the metrics
            #todo metric 바꾸기
            if CFG.wandb:
                wandb.log({"Train Loss": train_loss, 
                        "Valid Loss": val_loss,
                        "Valid PSNR": val_psnr,
                        "LR":scheduler.get_last_lr()[0]})
            
            LOGGER.info(f'{epoch}Epoch | Valid Loss: {val_loss:0.4f} | Valid PSNR: {val_psnr:0.2f} | LR: {scheduler.get_last_lr()[0]:0.7f}')
            
            # deep copy the model
            if val_psnr > best_psnr:
                LOGGER.info(f"Valid PSNR Improved ({best_psnr:0.4f} ---> {val_psnr:0.4f})")
                best_psnr    = val_psnr
                best_epoch   = epoch
                if CFG.wandb:
                    run.summary["Best PSNR"] = val_psnr
                    run.summary["Best Epoch"]   = best_epoch

                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR,f'fold{fold}_best.pth'))
                
                
            print(); print()

            if CFG.save_img:
                save_img(y_pred, epoch, comment='val')
                save_img(y_pred2, epoch, comment='skt_test')

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        else:
            print('else')
            scheduler.step()
    end = time.time()
    time_elapsed = end - start
    LOGGER.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    LOGGER.info(f"{best_epoch}Epoch | Best Score: {best_psnr:.2f}")
    
    # load best model weights
    #model.load_state_dict(best_model_wts)
    
    return best_psnr



def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


# ------------------------
#  Main
# ------------------------
def main(CFG):
    if CFG.wandb:
        wandb.login(key=CFG.wb_key)
    """
    Prepare: 1.train 
    """
    fold_score={}

    df = load_data(CFG)
    skt_df = pd.read_csv(f'./data/skt_test/HR_{CFG.img_size}/train.csv')


    for encoder in CFG.backbone:
        for decoder in CFG.model_name:
            fold_score[f'{encoder}_{decoder}']=[]
            for fold in CFG.folds:
                print(f'#'*15)
                print(f'### Fold: {fold}')
                print(f'#'*15)
                CFG.comment = f'{decoder}-{encoder}-{CFG.add_comment}'

                # related of saved ########################################
                CFG.OUTPUT_DIR = f'./saved_models/{CFG.comment}' 
                if not os.path.exists(CFG.OUTPUT_DIR):
                    os.makedirs(CFG.OUTPUT_DIR)

                # logger
                LOGGER = init_logger(log_file=os.path.join(CFG.OUTPUT_DIR, f'train_{fold}.log'), name=f'train{fold}_{CFG.comment}')
                ###########################################################
                run=None
                if CFG.wandb:
                    run = wandb.init(project='SKT_colorization', 
                                     config=class2dict(CFG),
                                     name=f"fold-{fold}|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{decoder}",
                                     group=CFG.comment,
                                    )

                # model
                model = build_model(CFG, encoder=encoder, decoder=decoder)
                if len(CFG.gpu)>1:
                    model = nn.DataParallel(model)
                    model.to(CFG.device)
                else:
                    model.to(CFG.device)
                
                if CFG.pretrained != 'None':
                    model.load_state_dict(torch.load(CFG.pretrained))


                # run train
                best_dice_score = run_training(model, df, skt_df, run, CFG.device, fold, LOGGER)
                
                fold_score[f'{encoder}_{decoder}'].append({f'{fold}fold' : best_dice_score})
                
                if CFG.wandb:
                    wandb.finish()
    
    return fold_score

if __name__ == '__main__':
    fold_score = main(CFG)
    print(fold_score)
