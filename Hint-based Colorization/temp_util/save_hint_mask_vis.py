import pandas as pd
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os



def save_fig(txt_list1, size=0, save=False):
    for txt_ in txt_list1:
        fn = txt_.split('/')[-1].split('.')[0]
        df = pd.read_csv(txt_, names=['pixel'])
        df['h'] = df['pixel'].apply(lambda x: x.split()[0]).astype('int')
        df['w'] = df['pixel'].apply(lambda x: x.split()[1]).astype('int')
        
        # 이미지 폴더명 변경..
#         img = cv2.imread(f'./data/gt/{fn}.JPEG')
        # img = cv2.imread(f'./data/518_2/{fn}.jpg')
        img = cv2.imread(f'/home/data/imagenet/ctest10k/{fn}.JPEG')



        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(224,224))
        mask = np.zeros((224,224,1))
        for i, j in df[['h', 'w']].values:
            mask[i:i+4, j:j+4, :] = 1

        masked_img = (img*mask).astype(np.uint8)
        masked_img[masked_img==0]=255
        # plt.imshow(img)
        plt.figure()
        plt.imshow(masked_img, alpha=1)
        plt.xticks([], [])
        plt.yticks([], [])

        if save:
            os.makedirs(f'./sample/ctest10k_hint_mask/h2-n{size}', exist_ok=True)
            plt.savefig(f'./sample/ctest10k_hint_mask/h2-n{size}/{fn}.jpg')
    

# save_fig(glob('./data/gt_txt/1234/h2-n1/*'), size=1, save=True)
# save_fig(glob('./data/gt_txt/1234/h2-n5/*'), size=5, save=True)
# save_fig(glob('./data/gt_txt/1234/h2-n10/*'), size=10, save=True)
# save_fig(glob('./data/gt_txt/1234/h2-n50/*'), size=50, save=True)

save_fig(glob('./data/ctest10k/1234/h2-n1/*'), size=1, save=True)
save_fig(glob('./data/ctest10k/1234/h2-n5/*'), size=5, save=True)
save_fig(glob('./data/ctest10k/1234/h2-n10/*'), size=10, save=True)
save_fig(glob('./data/ctest10k/1234/h2-n50/*'), size=50, save=True)