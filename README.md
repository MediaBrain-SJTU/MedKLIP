# MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology

## Introduction: 

The official implementation  code for "MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training in Radiology".

[**Paper Web**](https://chaoyi-wu.github.io/MedKLIP/) 

[**Arxiv Version**](https://arxiv.org/abs/2301.02228)

## Quick Start:
Check checkpoints dir to download our pre-trained model. It can be used for all zero-shot && finetuning tasks 

* **Zero-Shot Classification:**
    
    We give an example on CXR14 in ```Sample_Zero-Shot_Classification_CXR14```. Modify the path, and test our model by ```python test.py```
* **Zero-Shot Grounding:**
    
    We give an example on RSNA_Pneumonia in ```Sample_Zero-Shot_Grounding_RSNA```. Modify the path, and test our model by ```python test.py```
* **Finetuning:**
    
    We give segmentation and classification finetune code on SIIM_ACR dataset in ```Sample_Finetuning_SIIMACR```. Modify the path, and finetune our model by ```python I1_classification/train_res_ft.py``` or ```python I2_segementation/train_res_ft.py```

## Pre-train:
Our pre-train code is given in ```Train_MedKLIP```. 
* Check the ```Train_MedKLIP\data_file``` dir and download the pre-processed data files. 
* Modify the path as you disire, and ```python PreTrain_MedKLIP\train_MedKLIP.py``` to pre-train.

## Acknowledge
We borrow some pre-process codes from [AGXnet](https://github.com/batmanlab/AGXNet)

## Citation
```
@article{wu2023medklip,
  title={MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training},
  author={Wu, Chaoyi and Zhang, Xiaoman and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  journal={arXiv preprint arXiv:2301.02228},
  year={2023}
}
```
## Contact
If you have any question, please feel free to contact wtzxxxwcy02@sjtu.edu.cn.
