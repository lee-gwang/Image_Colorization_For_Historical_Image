from albumentations.pytorch import ToTensorV2
import albumentations as A
#from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage import color
from PIL import Image
HR_data_transforms = {
    "train": A.Compose([
        # A.RandomCrop(768,768),
        # A.Resize(size_,size_),
        A.HorizontalFlip(p=0.5),

        # A.RandomResizedCrop(224,224, p=1),

        A.GaussNoise(p=0.3),
        A.CoarseDropout (max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, 
                        min_width=8, fill_value=0, mask_fill_value=None, p=0.3),
        # A.RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True)
        ], p=1.0),
    
    # validation은 모든 패치를 이루도록?
    "valid": A.Compose([
        # A.RandomCrop(768,768),
        # A.Resize(768,768),
        ToTensorV2(transpose_mask=True)
        ], p=1.0),

    "infer": A.Compose([
        # A.RandomCrop(size_,size_),
        A.Resize(1024,1024),
        ToTensorV2(transpose_mask=True)
        ], p=1.0),

    "patch_infer": A.Compose([
        # A.RandomCrop(size_,size_),
        # A.RandomBrightness(limit=[-0.1,-0.1], p=1),
        ToTensorV2(transpose_mask=True)
        ], p=1.0),


        }


# opencv
class ActivityDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, type='train'):
        self.df         = df
        self.label      = label
        #self.LR_rgb_paths  = df['LR_rgb_paths'].tolist()
        self.HR_rgb_paths = df['new_rgb_paths'].tolist()
        self.hr_transforms = HR_data_transforms[type]
        #self.lr_transforms = LR_data_transforms[type]

        # norm
        self.l_cent = 50. #
        self.l_norm = 100. #
        self.ab_norm = 110.  #-128 to +127

    def norm(self, x, type = 'L'):
        if type =='L':
            #x[:,:,0] = (x[:,:,0] - 50.)/100.
            x[:,:,0] = (x[:,:,0] - 50.) / 100.
        elif type == 'AB':
            x[:,:,1:] = x[:,:,1:] / 110.

        return x
        
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
        # lab_input_img = self.norm(lab_input_img.astype(np.float32), type='L')[:,:,0]
        # lab_label_img = self.norm(self.norm(lab_label_img.astype(np.float32), type='L'), type='AB')


        if self.label:
            data = self.hr_transforms(image=lab_input_img, mask=lab_label_img)
            # lab_input_img = data['image']
            # lab_label_img = data['mask']

            lab_label_img  = data['mask']/255. # HR
            lab_input_img  = data['image']/255. # LR

            #data = self.lr_transforms(image=data['image'], mask=data['mask'])

                
            return lab_input_img, lab_label_img

        else: 
            data = self.hr_transforms(image=lab_input_img)
            lab_input_img  = data['image']/255. # LR
            return lab_input_img, hr_rgb_path.split('/')[-1] # filename        

# 0827 opencv2 / 
class ActivityDataset2(torch.utils.data.Dataset):
    def __init__(self, df, label=True, type='train'):
        self.df         = df
        self.label      = label
        #self.LR_rgb_paths  = df['LR_rgb_paths'].tolist()
        self.HR_rgb_paths = df['new_rgb_paths'].tolist()
        HR_data_transforms = {
                "train": A.Compose([
                    A.RandomResizedCrop(224,224, p=1),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                
                # validation은 모든 패치를 이루도록?
                "valid": A.Compose([
                    A.Resize(224,224),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "infer": A.Compose([
                    # A.RandomCrop(size_,size_),
                    A.Resize(1024,1024),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "patch_infer": A.Compose([
                    # A.Resize(224,224),
                    # A.RandomBrightness(limit=[-0.1,-0.1], p=1),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                    }

        self.hr_transforms = HR_data_transforms[type]
        #self.lr_transforms = LR_data_transforms[type]

        # norm
        self.l_cent = 50. #
        self.l_norm = 100. #
        self.ab_norm = 110.  #-128 to +127

    def norm(self, x, type = 'L'):
        if type =='L':
            #x[:,:,0] = (x[:,:,0] - 50.)/100.
            x[:,:,0] = (x[:,:,0] - 50.) / 100.
        elif type == 'AB':
            x[:,:,1:] = x[:,:,1:] / 110.

        return x
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load hr rgb images
        hr_rgb_path  = self.HR_rgb_paths[index]
        hr_rgb_img = cv2.imread(hr_rgb_path)
        # HR LAB label
        lab_label_img = cv2.cvtColor(hr_rgb_img, cv2.COLOR_BGR2LAB)

        # gray -> LAB
        lab_input_img = lab_label_img[:,:,0]

        if self.label:
            data = self.hr_transforms(image=lab_input_img, mask=lab_label_img)
            # lab_input_img = data['image']
            # lab_label_img = data['mask']

            lab_label_img  = data['mask']/255. # HR
            lab_input_img  = data['image']/255. # LR

            #data = self.lr_transforms(image=data['image'], mask=data['mask'])

                
            return lab_input_img, lab_label_img

        else: 
            data = self.hr_transforms(image=lab_input_img)
            lab_input_img  = data['image']/255. # LR
            return lab_input_img       


# 0820 new
class ColorDataset(torch.utils.data.Dataset):

    def __init__(self, df, label=True, type='train'):
        self.df         = df
        self.label      = label
        self.HR_rgb_paths = df['new_rgb_paths'].tolist()
        HR_data_transforms = {
            "train": A.Compose([
                # A.RandomCrop(768,768),
                # A.Resize(size_,size_),
                # A.HorizontalFlip(p=0.5),

                A.RandomResizedCrop(224,224, p=1),

                # A.GaussNoise(p=0.3),
                # A.CoarseDropout (max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, 
                #                 min_width=8, fill_value=0, mask_fill_value=None, p=0.3),
                # A.RandomRotate90(p=0.5),
                ToTensorV2(transpose_mask=True)
                ], p=1.0),
            
            # validation은 모든 패치를 이루도록?
            "valid": A.Compose([
                # A.RandomCrop(768,768),
                A.Resize(224,224),
                ToTensorV2(transpose_mask=True)
                ], p=1.0),

            "infer": A.Compose([
                # A.RandomCrop(size_,size_),
                A.Resize(1024,1024),
                ToTensorV2(transpose_mask=True)
                ], p=1.0),

            "patch_infer": A.Compose([
                # A.Resize(384,384, interpolation=cv2.INTER_AREA),
                # A.RandomBrightness(limit=[-0.1,-0.1], p=1),
                ToTensorV2(transpose_mask=True)
                ], p=1.0),


                }


        self.hr_transforms = HR_data_transforms[type]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load hr rgb images

        img = Image.open(self.df['new_rgb_paths'].loc[index]).convert('RGB')
        img = np.array(img)
        lab = color.rgb2lab(img).astype(np.float32) # 2ch
        x = lab[:,:,[0]]
        y = lab[:,:,[1,2]]

        # transform
        if self.label:
            data = self.hr_transforms(image=x, mask=y)

            x = data['image'] / 50.0 - 1.0 # [-1, 1]
            y = data['mask'] / 100.0 # [-1, 1]

            return x,y

        else:
            data = self.hr_transforms(image=x)

            x = data['image'] / 50.0 - 1.0 # [-1, 1]       

            return x


# 0901 new, 찐으로 제대로
class ImagenetDataset(torch.utils.data.Dataset):

    def __init__(self, df, label=True, type='train'):
        self.df         = df
        self.label      = label
        self.HR_rgb_paths = df['new_rgb_paths'].tolist()
        HR_data_transforms = {
                "train": A.Compose([
                    A.HorizontalFlip(p=0.5),    
                    A.RandomResizedCrop(512, 512, p = 0.3),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                
                # validation은 모든 패치를 이루도록?
                "valid": A.Compose([
                    # A.Resize(384,384),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "infer": A.Compose([
                    # A.RandomCrop(size_,size_),
                    A.Resize(1024,1024),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "patch_infer": A.Compose([
                    # A.Resize(224,224),
                    # A.RandomBrightness(limit=[-0.1,-0.1], p=1),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                    }


        
        self.hr_transforms = HR_data_transforms[type]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load hr rgb images

        img = Image.open(self.df['new_rgb_paths'].loc[index]).convert('RGB')
        img = color.rgb2lab(img).astype(np.float32)
        # img = np.array(img) # 3ch

        # transform
        if self.label:
            data = self.hr_transforms(image=img)

            return data['image']#/255.

        else:
            data = self.hr_transforms(image=img)

            return data['image']#/255.

# 0829 , imagenet, pillow
# 이거쓸때는 train2_imagenet에서 rgb2lab 쓰기
class ImagenetDataset3(torch.utils.data.Dataset):

    def __init__(self, df, label=True, type='train'):
        self.df         = df
        self.label      = label
        self.HR_rgb_paths = df['new_rgb_paths'].tolist()
        HR_data_transforms = {
                "train": A.Compose([
                    A.RandomResizedCrop(224,224, p=0.4),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                
                # validation은 모든 패치를 이루도록?
                "valid": A.Compose([
                    A.Resize(224,224),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "infer": A.Compose([
                    # A.RandomCrop(size_,size_),
                    A.Resize(1024,1024),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "patch_infer": A.Compose([
                    A.Resize(224,224),
                    # A.RandomBrightness(limit=[-0.1,-0.1], p=1),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                    }


        
        self.hr_transforms = HR_data_transforms[type]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load hr rgb images

        img = Image.open(self.df['new_rgb_paths'].loc[index]).convert('RGB')
        img = np.array(img)
        lab = color.rgb2lab(img).astype(np.float32) # 2ch
        x = lab[:,:,[0]]
        y = lab[:,:,[1,2]]

        # transform
        if self.label:
            data = self.hr_transforms(image=x, mask=y)

            x = data['image'] / 50.0 - 1.0 # [-1, 1]
            y = data['mask'] / 100.0 # [-1, 1]

            return x,y

        else:
            data = self.hr_transforms(image=x)

            x = data['image'] / 50.0 - 1.0 # [-1, 1]       

            return x

# 0830 , imagenet - opencv version
# 이거쓸때는 train2_imagenet에서 255 곱하는걸로만쓰기
# save image 함수도 바꾸기
class ImagenetDataset2(torch.utils.data.Dataset):
    def __init__(self, df, label=True, type='train'):
        self.df         = df
        self.label      = label
        self.HR_rgb_paths = df['new_rgb_paths'].tolist()
        HR_data_transforms = {
                "train": A.Compose([
                    A.RandomResizedCrop(384,384, p=0.4),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                
                # validation은 모든 패치를 이루도록?
                "valid": A.Compose([
                    A.Resize(384,384),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "infer": A.Compose([
                    # A.RandomCrop(size_,size_),
                    A.Resize(1024,1024),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),

                "patch_infer": A.Compose([
                    # A.Resize(224,224),
                    # A.RandomBrightness(limit=[-0.1,-0.1], p=1),
                    ToTensorV2(transpose_mask=True)
                    ], p=1.0),
                    }
        self.hr_transforms = HR_data_transforms[type]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load hr rgb images
        hr_rgb_path  = self.HR_rgb_paths[index]
        hr_rgb_img = cv2.imread(hr_rgb_path)
        # HR LAB label
        lab_label_img = cv2.cvtColor(hr_rgb_img, cv2.COLOR_BGR2LAB)

        # gray -> LAB
        lab_input_img = lab_label_img[:,:,0]
        lab_label_img = lab_label_img[:,:,[1,2]]

        if self.label:
            data = self.hr_transforms(image=lab_input_img, mask=lab_label_img)

            lab_input_img  = data['image']/255. # LR
            lab_label_img  = data['mask']/255. # HR

                
            return lab_input_img, lab_label_img

        else: 
            data = self.hr_transforms(image=lab_input_img)
            lab_input_img  = data['image']/255. # LR
            return lab_input_img       