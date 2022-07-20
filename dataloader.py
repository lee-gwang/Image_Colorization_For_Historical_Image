from albumentations.pytorch import ToTensorV2
import albumentations as A
#from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2

data_transforms = {
"train": A.Compose([
    A.HorizontalFlip(p=0.5),

    ToTensorV2(transpose_mask=True)
    ], p=1.0),

"valid": A.Compose([
    ToTensorV2(transpose_mask=True)
    ], p=1.0)
    }

size_ = 384
HR_data_transforms = {
    "train": A.Compose([
        #A.RandomCrop(size_,size_),
        # A.Resize(size_,size_),
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True)
        ], p=1.0),
    
    # validation은 모든 패치를 이루도록?
    "valid": A.Compose([
        # A.RandomCrop(size_,size_),
        # A.Resize(size_,size_),
        ToTensorV2(transpose_mask=True)
        ], p=1.0)
        }
# LR_data_transforms = {
#     "train": A.Compose([
#         A.Resize(size_//4,size_//4),
#         ToTensorV2(transpose_mask=True)
#         ], p=1.0),
    
#     # validation은 모든 패치를 이루도록?
#     "valid": A.Compose([
#         A.Resize(size_//4, size_//4),
#         ToTensorV2(transpose_mask=True)
#         ], p=1.0)
#         }
"""
오직 L만 들어가야함
"""
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df         = df
        self.label      = label
        self.rgb_paths  = df['rgb_paths'].tolist()
        self.transforms = data_transforms
        
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

class ActivityDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, type='train'):
        self.df         = df
        self.label      = label
        #self.LR_rgb_paths  = df['LR_rgb_paths'].tolist()
        self.HR_rgb_paths = df['new_rgb_paths'].tolist()
        self.hr_transforms = HR_data_transforms[type]
        #self.lr_transforms = LR_data_transforms[type]

        # norm
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load hr rgb images
        hr_rgb_path  = self.HR_rgb_paths[index]
        hr_rgb_img = cv2.imread(hr_rgb_path)
        # HR LAB label
        lab_label_img = cv2.cvtColor(hr_rgb_img, cv2.COLOR_BGR2LAB)

        # rgb -> gray
        hr_gray_img = cv2.cvtColor(hr_rgb_img, cv2.COLOR_BGR2GRAY)
        hr_gray_img = np.stack([hr_gray_img, hr_gray_img, hr_gray_img], -1)

        # gray -> LAB
        lab_input_img = cv2.cvtColor(hr_gray_img, cv2.COLOR_BGR2LAB)[:,:,0]


        if self.label:
            data = self.hr_transforms(image=lab_input_img, mask=lab_label_img)
            lab_label_img  = data['mask']/255. # HR
            lab_input_img  = data['image']/255. # LR

            #data = self.lr_transforms(image=data['image'], mask=data['mask'])

                
            return lab_input_img, lab_label_img

        else: 
            data = self.hr_transforms(image=lab_input_img)
            lab_input_img  = data['image']/255. # LR
            return lab_input_img       
