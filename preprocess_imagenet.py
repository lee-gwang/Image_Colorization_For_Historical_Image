import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import os
import cv2

df = pd.DataFrame()
df['rgb_paths'] = glob.glob('/home/data/imagenet/train/*/*')
df['class'] = df['rgb_paths'].apply(lambda x:x.split('/')[5])


new_df = df.groupby('class').sample(200).reset_index(drop=True)

# make folders
for i in glob.glob('/home/data/imagenet/train/*'):
    os.makedirs(i.replace('imagenet','imagenet20k'))
    
    
# preprocess images
from tqdm import tqdm
for p in tqdm(new_df['rgb_paths']):
    img = cv2.imread(p)
    img = cv2.resize(img, (224,224))
    new_path = p.replace('imagenet','imagenet20k')
    cv2.imwrite(new_path, img)
    
os.makedirs('./data/imagenet_sample20k')


# save csv
skf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(new_df)):
    new_df.loc[val_idx, 'fold'] = fold
new_df['rgb_paths'] = new_df['rgb_paths'].apply(lambda x: x.replace('imagenet', 'imagenet20k'))   
new_df.to_csv('./data/imagenet_sample20k/train.csv', index=False)