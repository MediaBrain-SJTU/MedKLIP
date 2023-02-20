import argparse
import os
import ruamel_yaml as yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resunet import ModelResUNet_ft

from dataset.dataset_siim_acr import SIIM_ACR_Dataset
from metric import mIoU,dice



def test(args,config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    test_dataset =  SIIM_ACR_Dataset(config['test_file'],is_train = False) 
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )              
    
    model = ModelResUNet_ft(res_base_model='resnet50',out_size=1,imagenet_pretrain=False)
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device) 

    print('Load model from checkpoint:',args.model_path)
    checkpoint = torch.load(args.model_path,map_location='cpu') 
    state_dict = checkpoint['model']          
    model.load_state_dict(state_dict)    

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    print("Start testing")
    model.eval()
    for i, sample in enumerate(test_dataloader):
        image = sample['image']
        mask = sample['seg'].float().to(device)
        gt = torch.cat((gt, mask), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_mask = model(input_image)
            pred_mask = F.sigmoid(pred_mask)
            pred = torch.cat((pred, pred_mask), 0)
    dice_score, dice_neg, dice_pos, num_neg, num_pos = dice(pred,gt)
    IoU_score = mIoU(pred, gt) 
    print('Dice score is', dice_score)
    print('IoU score is', IoU_score)
    return dice_score,IoU_score
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='Path/To/Res_train.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='Path/To/best_valid.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='0', help='gpu')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    test(args, config)
