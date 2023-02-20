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
# from dataset.randaugment import RandomAugment
from torchvision.transforms import InterpolationMode
class RSNA2018_Dataset(Dataset):
    def __init__(self, csv_path):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,1])
        self.class_list = np.asarray(data_info.iloc[:,3])
        self.bbox = np.asarray(data_info.iloc[:,2])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

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
        img_path = self.img_path_list[index]
        class_label = np.array([self.class_list[index]]) # (14,)

        img = self.read_dcm(img_path) 
        image = self.transform(img)
        
        bbox = self.bbox[index]
        seg_map = np.zeros((1024,1024))
        if class_label ==1:
            boxes = bbox.split('|')
            for box in boxes:
                cc = box.split(';')
                seg_map[int(float(cc[1])):(int(float(cc[1]))+int(float(cc[3]))),int(float(cc[0])):(int(float(cc[0]))+int(float(cc[2])))]=1
        seg_map = self.seg_transfrom(seg_map)
        return {
            "image": image,
            "label": class_label,
            "image_path": img_path,
            "seg_map":seg_map
            }
    
    def read_dcm(self,dcm_path):
        dcm_data = pydicom.read_file(dcm_path)
        img = dcm_data.pixel_array.astype(float) / 255.
        img = exposure.equalize_hist(img)
        
        img = (255 * img).astype(np.uint8)
        img = PIL.Image.fromarray(img).convert('RGB')   
        return img


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

