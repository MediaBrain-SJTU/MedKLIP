import argparse
import os
import yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score

from models.model_MedKLIP import MedKLIP
from dataset.dataset import Chestxray14_Dataset
from models.tokenization_bert import BertTokenizer

chexray14_cls = [ 'atelectasis', 'cardiomegaly', 'effusion', 'infiltrate', 'mass', 'nodule', 'pneumonia',
                'pneumothorax', 'consolidation', 'edema', 'emphysema', 'tail_abnorm_obs', 'thicken', 'hernia']  #Fibrosis seldom appears in MIMIC_CXR and is divided into the 'tail_abnorm_obs' entitiy.  

original_class = [
            'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
            'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
            'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
            'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
            'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
            'tail_abnorm_obs', 'excluded_obs'
        ]

mapping = []
for disease in chexray14_cls:
    if disease in original_class:
        mapping.append(original_class.index(disease))
    else:
        mapping.append(-1)
MIMIC_mapping = [ _ for i,_ in enumerate(mapping) if _ != -1]
chexray14_mapping = [ i for i,_ in enumerate(mapping) if _ != -1]
target_class = [ chexray14_cls[i] for i in chexray14_mapping ]

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def test(args,config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    test_dataset =  Chestxray14_Dataset(config['test_file'],is_train=False) 
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=False,
        ) 
                 
    
    print("Creating book")
    json_book = json.load(open(config['disease_book'],'r'))
    disease_book = [json_book[i] for i in json_book]
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)
    
    print("Creating model")
    model = MedKLIP(config, disease_book_tokenizer)
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)  

    print('Load model from checkpoint:',args.model_path)
    checkpoint = torch.load(args.model_path,map_location='cpu') 
    state_dict = checkpoint['model']          
    model.load_state_dict(state_dict)    

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.to(device)
    pred = torch.FloatTensor()
    pred = pred.to(device)
    
    print("Start testing")
    model.eval()
    for i, sample in enumerate(test_dataloader):
        image = sample['image']
        label = sample['label'][:,chexray14_mapping].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_class = model(input_image) #batch_size,num_class,dim
            pred_class = F.softmax(pred_class.reshape(-1,2)).reshape(-1,len(original_class),2)
            pred_class = pred_class[:,MIMIC_mapping,1]
            pred = torch.cat((pred, pred_class), 0)
    
    AUROCs = compute_AUCs(gt, pred,len(target_class))
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
    for i in range(len(target_class)):
        print('The AUROC of {} is {}'.format(target_class[i], AUROCs[i]))
    max_f1s = []
    accs = []
    for i in range(len(target_class)):   
        gt_np = gt[:, i].cpu().numpy()
        pred_np = pred[:, i].cpu().numpy()  
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))
        
    f1_avg = np.array(max_f1s).mean()    
    acc_avg = np.array(accs).mean()
    print('The average f1 is {F1_avg:.4f}'.format(F1_avg=f1_avg))
    print('The average ACC is {ACC_avg:.4f}'.format(ACC_avg=acc_avg))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/Path/To/MedKLIP_config.yaml')
    parser.add_argument('--model_path', default='/Path/To/checkpoint.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='1', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != '-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    test(args, config)

