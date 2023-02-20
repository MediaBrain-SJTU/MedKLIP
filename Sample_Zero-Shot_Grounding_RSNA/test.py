
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset_RSNA import RSNA2018_Dataset
from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer

original_class = [
            'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
            'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
            'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
            'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
            'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
            'tail_abnorm_obs', 'excluded_obs'
        ]
def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def score_cal(labels,seg_map,pred_map):
    '''
    labels B * 1
    seg_map B *H * W
    pred_map B * H * W
    '''
    device = labels.device
    total_num = torch.sum(labels)
    mask = (labels==1).squeeze()
    seg_map = seg_map[mask,:,:].reshape(total_num,-1)
    pred_map = pred_map[mask,:,:].reshape(total_num,-1)
    one_hot_map = (pred_map > 0.008)
    dot_product = (seg_map *one_hot_map).reshape(total_num,-1)
    
    max_number = torch.max(pred_map,dim=-1)[0]
    point_score = 0
    for i,number in enumerate(max_number):
        temp_pred = (pred_map[i] == number).type(torch.int)
        flag = int((torch.sum(temp_pred * seg_map[i]))>0)
        point_score = point_score + flag
    mass_score = torch.sum(dot_product,dim = -1)/((torch.sum(seg_map,dim=-1)+torch.sum(one_hot_map,dim=-1))-torch.sum(dot_product,dim = -1))
    dice_score = 2*(torch.sum(dot_product,dim=-1))/(torch.sum(seg_map,dim=-1)+torch.sum(one_hot_map,dim=-1))
    return total_num,point_score,mass_score.to(device),dice_score.to(device)


def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    #### Dataset #### 
    print("Creating dataset")
    test_dataset =  RSNA2018_Dataset(config['test_file']) 
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=0,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )              
    json_book = json.load(open(config['disease_book'],'r'))
    disease_book = [json_book[i] for i in json_book]
    ana_book = [ 'It is located at ' + i for i in ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
            'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
            'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe',
            'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
            'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung', 'right_apical_lung',
            'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic', 'right_costophrenic', 'costophrenic_unspec',
            'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc',
            'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other']]
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    ana_book_tokenizer = get_tokenizer(tokenizer,ana_book).to(device)
    disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)
    
    print("Creating model")
    model = MedKLIP(config,ana_book_tokenizer, disease_book_tokenizer, mode = 'train')
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)  
    

    checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)    
    print('load checkpoint from %s'%args.checkpoint)

    print("Start testing")
    model.eval()

    dice_score_A = torch.FloatTensor()
    dice_score_A = dice_score_A.to(device)
    mass_score_A = torch.FloatTensor()
    mass_score_A = mass_score_A.to(device)
    total_num_A = 0
    point_num_A = 0

    
    for i, sample in enumerate(test_dataloader):
        images = sample['image'].to(device)
        image_path = sample['image_path']
        batch_size = images.shape[0]
        labels = sample['label'].to(device)
        seg_map = sample['seg_map'][:,0,:,:].to(device) #B C H W

        with torch.no_grad():
            _,ws= model(images,labels,is_train= False) #batch_size,batch_size,image_patch,text_patch
            ws = (ws[-4] +ws[-3]+ws[-2]+ws[-1])/4
            ws = ws.reshape(batch_size,ws.shape[1],14,14)
            pred_map = ws[:,original_class.index('pneumonia'),:,:].detach().cpu().numpy()
            
            pred_map = torch.from_numpy(pred_map.repeat(16, axis=1).repeat(16, axis=2)).to(device) #Final Grounding Heatmap
            
            total_num,point_num,mass_score,dice_score = score_cal(labels,seg_map,pred_map) 
            total_num_A = total_num_A+total_num
            point_num_A = point_num_A+point_num
            dice_score_A = torch.cat((dice_score_A,dice_score),dim=0)
            mass_score_A = torch.cat((mass_score_A,mass_score),dim=0)
            
                
    dice_score_avg = torch.mean(dice_score_A)
    mass_score_avg = torch.mean(mass_score_A)
    print('The average dice_score is {dice_score_avg:.5f}'.format(dice_score_avg=dice_score_avg))
    print('The average iou_score is {mass_score_avg:.5f}'.format(mass_score_avg=mass_score_avg))
    point_score = point_num_A/total_num_A
    print('The average point_score is {point_score:.5f}'.format(point_score=point_score))                      
            
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/Path/To/MedKLIP_config.yaml')
    parser.add_argument('--model_path', default='/Path/To/checkpoint.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='1', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu !='-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True

    main(args, config)