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


args = parser.parse_args()


def activity_preprocessing():
    df = pd.DataFrame()
    df['hr_rgb_paths'] = glob.glob('/home/data/activitynet/labels/*/*')
    def hw_cal(path):
        img = cv2.imread(path)
        h,w, _ = img.shape
        
        return h,w
        
    hw = Parallel(n_jobs=-1, backend='threading')(delayed(hw_cal)(path_)\
                                                for path_ in tqdm(df['hr_rgb_paths'].tolist(), total=len(df['hr_rgb_paths'])))

    df['h'] = [h for h,w in hw]
    df['w'] = [w for h,w in hw]


    # activity preprocessing
    """
    1. hw 구해서 480px 이상 8의 배수로만 셋팅
    """

    df = pd.DataFrame()
    df['hr_rgb_paths'] = glob.glob('/home/data/activitynet/labels/*/*')
    def hw_cal(path):
        img = cv2.imread(path)
        h,w, _ = img.shape
        
        return h,w
        
    hw = Parallel(n_jobs=-1, backend='threading')(delayed(hw_cal)(path_)\
                                                for path_ in tqdm(df['rgb_paths'].tolist(), total=len(df['rgb_paths'])))

    df['h'] = [h for h,w in hw]
    df['w'] = [w for h,w in hw]

    # 8의 배수이고, 480 이상인 해상도
    new_df = df[(df['h']%8 ==0)&(df['w']%8 ==0)&(df['h']>480)&(df['w']>480)].reset_index(drop=True)


    """
    2. resize해서 저장
    """
    # def resize_images(path,scale=4):
    #     """
    #     scale :
    #     ex) scale =4, downsize =1/4
    #     """
    #     img = cv2.imread(path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img=cv2.resize(img,dsize=(0,0), fx=1/scale, fy=1/scale)
        
    #     # save
    #     new_path = path.replace('labels', f'X{scale}')
    #     cv2.imwrite(new_path, img)
        
    #     return


    # scale_ = 4 # X4 downsampling

    # # make folders
    # for i in glob.glob('/home/data/activitynet/labels/*'):
    #     os.makedirs(i.replace('labels',f'X{scale_}'))

    # # save resized images
    # _ = Parallel(n_jobs=-1, backend='threading')(delayed(resize_images)(path_, scale_)\
    #                                              for path_ in tqdm(new_df['rgb_paths'].tolist(), total=len(new_df['rgb_paths'])))

    """
    3. train, test 나누기
    # groupkfold를 이용해서 videoid가 같을경우 겹치지 않게 하기
    # 나중에는 코드 안에서 X4, X8, X2 등으로 바꿀 수 있으니, 재실행 안해도됨
    """
    #
    os.makedirs('./data/activitynet', exist_ok=True)

    new_df['video_id'] = new_df['hr_rgb_paths'].apply(lambda x: x.split('/')[-2])
    # new_df['rgb_paths'] = new_df['rgb_paths'].apply(lambda x: x.replace('/labels/', '/X4/'))
    # new_df['label_paths'] = new_df['rgb_paths'].apply(lambda x: x.replace('/X4/', '/labels/'))
    # new_df = new_df.rename(columns={'rgb_paths':'LR_rgb_paths', 'label_paths':'HR_rgb_paths'})

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(new_df, new_df['video_id'], groups = new_df["video_id"])):
        new_df.loc[val_idx, 'fold'] = fold
    new_df.to_csv('./data/activitynet/train.csv', index=False)

def imagenet_preprocessing():
    df = pd.DataFrame()

    # path
    data_path = args.data_path
    if args.depth == 2:
        data_path = glob.glob(os.path.join(data_path, '*/*'))
    elif args.depth == 1:
        data_path = glob.glob(os.path.join(data_path, '*'))

    # preprocess the dataframe
    df['rgb_paths'] = glob.glob(data_path) # ('/home/data/imagenet/train/*/*'
    df['class'] = df['rgb_paths'].apply(lambda x:x.split('/')[-2])

    # sampling
    #new_df = df.groupby('class').sample(200).reset_index(drop=True)

    # make folders
    for i in glob.glob('/home/data/imagenet/train/*'):
        os.makedirs(i.replace('imagenet','imagenet20k'))
        
        
    # preprocess images
    for p in tqdm(new_df['rgb_paths']):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.size,args.size))
        new_path = p.replace('imagenet','imagenet20k')
        cv2.imwrite(new_path, img)
        
    os.makedirs('./data/imagenet_sample20k')


    # save csv
    skf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(new_df)):
        new_df.loc[val_idx, 'fold'] = fold
    new_df['rgb_paths'] = new_df['rgb_paths'].apply(lambda x: x.replace('imagenet', 'imagenet20k'))   
    new_df.to_csv('./data/imagenet_sample20k/train.csv', index=False)



# os.system(f'ln -s {args.data_path} ./data/{args.data}/HR')
# os.makedirs(f'./data/{args.data}/LR', exist_ok=True)
# os.makedirs(f'./data/{args.data}/HR', exist_ok=True)
def resize(new_rgb_path, rgb_path):
    img = cv2.imread(rgb_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.size,args.size))

    img_name = new_rgb_path.split('/')[-1]
    new_folder = '/'.join(new_rgb_path.split('/')[:-1])
    os.makedirs(new_folder, exist_ok=True)
    new_path = os.path.join(new_folder, img_name)
    cv2.imwrite(new_path, img)
    return

def image_preprocessing(args):
    df = pd.DataFrame()

    # path
    data_path = args.data_path
    if args.depth == 2:
        data_path = glob.glob(os.path.join(data_path, '*/*'))
    elif args.depth == 1:
        data_path = glob.glob(os.path.join(data_path, '*'))

    # preprocess the dataframe
    df['rgb_paths'] = data_path # ('/home/data/imagenet/train/*/*'
    df['class'] = df['rgb_paths'].apply(lambda x:x.split('/')[-2])

    # saved new path
    if args.depth ==2:
        depth = -args.depth 
    df['new_rgb_paths'] = df['rgb_paths'].apply(lambda x: os.path.join(f'./data/{args.data}', # dataset name
                                                                        f'HR_{args.size}', # dataset size
                                                                        '/'.join(x.split('/')[depth:-1]), # class 
                                                                        x.split('/')[-1]))  # image name

    # preprocess images
    Parallel(n_jobs=-1, backend='threading')(delayed(resize)(new_path, path)\
                                for new_path, path in tqdm(df[['new_rgb_paths','rgb_paths']].values, total=len(df['rgb_paths'])))
    # save csv
    skf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df)):
        df.loc[val_idx, 'fold'] = fold
      
    df.to_csv(f'./data/{args.data}/HR_{args.size}/train.csv', index=False)

if __name__ == '__main__':
    image_preprocessing(args)
