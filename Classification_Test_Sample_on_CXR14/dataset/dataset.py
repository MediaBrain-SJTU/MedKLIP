import csv
import json
import logging
import os
import sys
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

from dataset.randaugment import RandomAugment


class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path,is_train = True):
        data_info = pd.read_csv(csv_path)
        # self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,3:])
        
        if is_train==True:
            total_len = int(0.01*len(data_info))
            choice_list = np.random.choice(range(len(data_info)), size = total_len,replace= False)
            self.img_path_list = np.asarray(data_info.iloc[:,0])[choice_list]
            self.class_list = np.asarray(data_info.iloc[:,3:])[choice_list]
        else:
            self.img_path_list = np.asarray(data_info.iloc[:,0])
            self.class_list = np.asarray(data_info.iloc[:,3:])
            

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

    def __getitem__(self, index):
        img_path = self.img_path_list[index].replace('/mnt/cfs/xmzhang/DATA/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        class_label = self.class_list[index]
        img = PIL.Image.open(img_path).convert('RGB')   
        image = self.transform(img)

        return {
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
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
