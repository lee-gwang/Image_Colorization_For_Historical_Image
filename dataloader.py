from albumentations.pytorch import ToTensorV2
import albumentations as A
#from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

"""
오직 L만 들어가야함
"""
class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df         = df
        self.label      = label
        self.rgb_paths  = df['rgb_paths'].tolist()
        self.transforms = transforms
        
        # norm
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # rgb images
        rgb_path  = self.rgb_paths[index]
        rgb_img = cv2.imread(rgb_path)
        #rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        gray_img = np.stack([gray_img, gray_img, gray_img], -1)

        lab_input_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2LAB)[:,:,0]
        lab_label_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)

        # l norm
        # lab_img[:,:,0] = (lab_img[:,:,0]-self.l_cent)/self.l_norm
        # # ab norm
        # lab_img[:,:,1:] = lab_img[:,:,1:]/self.ab_norm

        if self.label:
            if self.transforms:
                data = self.transforms(image=lab_input_img, mask=lab_label_img)
                lab_input_img  = data['image']/255.#/255.
                lab_label_img  = data['mask']/255.
            return lab_input_img, lab_label_img

        else:
            if self.transforms:
                data = self.transforms(image=lab_input_img)
                lab_input_img  = data['image']#/255.
            return lab_input_img

       

