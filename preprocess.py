import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Script of SKT Colorization')
parser.add_argument('--data', '-data', type=str, default='gettyimages', help='dataset name')
parser.add_argument('--data_path', '-dp', type=str, default='/home/data/imagenet/train', help='dataset path')
parser.add_argument('--size', '-s', type=int, default=224)
parser.add_argument('--depth', '-d', type=int, default=2, help='path depth')
parser.add_argument('--fold', type=int, default=5, help='fold')
parser.add_argument('--filtering', action='store_true', help='resize filtering images')



args = parser.parse_args()

def resize(new_rgb_path, rgb_path):
    img = cv2.imread(rgb_path)
    try:
        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.size,args.size))

        img_name = new_rgb_path.split('/')[-1]
        new_folder = '/'.join(new_rgb_path.split('/')[:-1])
        os.makedirs(new_folder, exist_ok=True)
        new_path = os.path.join(new_folder, img_name)
        cv2.imwrite(new_path, img)

        return h, w
    except:
        print('error image path : ', rgb_path)
        return 0, 0

def image_preprocessing(args):
    """
    resize 후 사용하는 코드
    """
    df = pd.DataFrame()

    # path
    data_path = args.data_path
    if args.depth == 2:
        data_path = glob.glob(os.path.join(data_path, '*/*'))
    elif args.depth == 1:
        data_path = glob.glob(os.path.join(data_path, '*'))

    elif args.depth == -1: # Search all of the images
        # please delete the fruits2/train/bell pepper/Image_56.jpg
        data_path = os.path.join(data_path, '*')
        image_list = []
        image_formats = ['jpg', 'JPG', 'JPEG', 'jpeg', 'png']
        while len(glob.glob(data_path)):
            image_list.extend(glob.glob(data_path))
            data_path = os.path.join(data_path, '*')
            print(data_path)

        image_list = [x for x in image_list if x.split('.')[-1] in image_formats ]
        data_path = image_list

    # preprocess the dataframe
    df['rgb_paths'] = data_path # ('/home/data/imagenet/train/*/*'

    # saved new path
    #if args.depth ==2:
    depth = -args.depth 
    df['new_rgb_paths'] = df['rgb_paths'].apply(lambda x: os.path.join(f'./data/{args.data}', # dataset name
                                                                        f'HR_{args.size}', # dataset size
                                                                        '/'.join(x.split('/')[depth:-1]), # class 
                                                                        x.split('/')[-1]))  # image name

    # preprocess images
    resolution = Parallel(n_jobs=-1, backend='threading')(delayed(resize)(new_path, path)\
                                for new_path, path in tqdm(df[['new_rgb_paths','rgb_paths']].values, total=len(df['rgb_paths'])))

    # width & height
    if args.filtering:
        org = df.shape[0]
        df['h'] = [x[0] for x in resolution]
        df['w'] = [x[1] for x in resolution]
        df['hw'] = df['h'] * df['w']
        df = df[df['hw']>=args.size*args.size].reset_index(drop=True)
        print(f'filter out {org - df.shape[0]} / {org} images !!')


    # df = df[~df['new_rgb_paths'].isin(error_)].reset_index(drop=True)
    # save csv
    skf = KFold(n_splits=args.fold, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df)):
        df.loc[val_idx, 'fold'] = fold
      
    df.to_csv(f'./data/{args.data}/HR_{args.size}/train_{args.fold}.csv', index=False)

def image_preprocessing_noresize(args):
    """
    원본 이미지 사용하는 코드
    """
    df = pd.DataFrame()

    # path
    data_path = args.data_path
    if args.depth == 2:
        data_path = glob.glob(os.path.join(data_path, '*/*'))
    elif args.depth == 1:
        data_path = glob.glob(os.path.join(data_path, '*'))

    elif args.depth == -1:
        # 모든 depth... 코드 이상..
        data_path = glob.glob(os.path.join(data_path, '*/*'))
        data_path.extend(glob.glob(os.path.join(data_path, '*/*/*')))

    # preprocess the dataframe
    df['new_rgb_paths'] = data_path # ('/home/data/imagenet/train/*/*'

    # save csv
    skf = KFold(n_splits=args.fold, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df)):
        df.loc[val_idx, 'fold'] = fold
      
    df.to_csv(f'{args.data_path}/train_{args.fold}.csv', index=False)


if __name__ == '__main__':
    image_preprocessing(args)
    # image_preprocessing_noresize(args)