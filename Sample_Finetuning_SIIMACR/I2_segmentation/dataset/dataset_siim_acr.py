from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class SIIM_ACR_Dataset(Dataset):
    def __init__(self, csv_path,is_train=True):
        data_info = pd.read_csv(csv_path)
        if is_train==True:
            total_len = int(0.01*len(data_info))
            choice_list = np.random.choice(range(len(data_info)), size = total_len,replace= False)
            self.img_path_list = np.asarray(data_info.iloc[:,0])[choice_list]
        else:
            self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.img_root = '/remote-home/share/medical/public/SIIM-ACR/processed_images/'
        self.seg_root = '/remote-home/share/medical/public/SIIM-ACR/segmentation_masks/' # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        
        if is_train:
            self.aug = A.Compose([
            A.RandomResizedCrop(width=224, height=224, scale=(0.2, 1.0), always_apply = True, interpolation=Image.BICUBIC),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225],always_apply = True),
            ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
            A.Resize(width=224, height=224, always_apply = True),
            A.Normalize(mean = [0.485, 0.456, 0.406], std =  [0.229, 0.224, 0.225],always_apply = True),
            ToTensorV2()
            ])

    def __getitem__(self, index):
        img_path = self.img_root + self.img_path_list[index] + '.png'
        seg_path = self.seg_root + self.img_path_list[index] + '.gif' # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        img = np.array(PIL.Image.open(img_path).convert('RGB') ) 
        seg_map = np.array(PIL.Image.open(seg_path))[:,:,np.newaxis]
        
        augmented = self.aug(image=img, mask=seg_map)
        img, seg_map = augmented['image'], augmented['mask']
        seg_map = seg_map.permute(2, 0, 1)
        
        class_label = np.array([int(torch.sum(seg_map)>0)])
        return {
            "image": img,
            "seg": seg_map,
            "label": class_label
            }


    def __len__(self):
        return len(self.img_path_list)
    

def create_loader_RSNA(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
