import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class ModelRes_ft(nn.Module):
    def __init__(self, res_base_model,out_size,imagenet_pretrain=False,linear_probe=False):
        super(ModelRes_ft, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=imagenet_pretrain),
                            "resnet50": models.resnet50(pretrained=imagenet_pretrain)}
        resnet = self._get_res_basemodel(res_base_model)
        num_ftrs = int(resnet.fc.in_features)
        self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        self.res_out = nn.Linear(num_ftrs, out_size)


    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, img,linear_probe=False):
        x = self.res_features(img)
        x = x.squeeze()
        if linear_probe:
            return x
        else:
            x = self.res_out(x)
            return x