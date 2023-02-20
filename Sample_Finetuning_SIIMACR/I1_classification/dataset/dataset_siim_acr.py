from cmath import nan
import csv
import json
import logging
import os
import sys
import pydicom

from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
from skimage import exposure
import torch
from torchvision.transforms import InterpolationMode
from dataset.randaugment import RandomAugment
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
        
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if is_train:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(224,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])   
        else:
            self.transform = transforms.Compose([                        
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize,
            ])     
        
        self.seg_transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224],interpolation=InterpolationMode.NEAREST),
        ])

    def __getitem__(self, index):
        img_path = self.img_root + self.img_path_list[index] + '.png'
        seg_path = self.seg_root + self.img_path_list[index] + '.gif'    # We have pre-processed the original SIIM_ACR data, you may change this to fix your data
        img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)

        seg_map = PIL.Image.open(seg_path)
        seg_map = self.seg_transfrom(seg_map)
        seg_map = (seg_map > 0).type(torch.int)
        class_label = np.array([int(torch.sum(seg_map)>0)])
        return {
            "image": image,
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
