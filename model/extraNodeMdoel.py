
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.arcface import *
from model.extraNode import *

class EfficientNet_ExtraNode(nn.Module):
    def __init__(self, num_class, model_name='b0', pretrained=True):
        super().__init__()

        self.model_size = model_name
        self.backbone = self.getEfficientNet(model_name, pretrained, 512)
        self.classifier = nn.Linear(in_features=512, out_features=num_class)
        # self.arc = ArcMarginProduct(512, num_class)
        self.extraNode = ExtraNode(512, 1)
    
    def __str__(self) -> str:
        return f"EfficientNet{self.model_size}"

    def forward(self, x, label):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        classiify = self.classifier(x)
        extra = self.extraNode(x)
        
        result = torch.cat((classiify, extra), 1)

        return result

    def getEfficientNet(self, name: str, pretrained: bool, embeding_features):
        name = str.lower(name)
        if name == 'b0':
            net = models.efficientnet_b0(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=1280, out_features=embeding_features)
            return net             
        elif name == 'b1':
            net = models.efficientnet_b1(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=1280, out_features=embeding_features)
            return net
        elif name == 'b2':
            net = models.efficientnet_b2(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=1408, out_features=embeding_features)
            return net
        elif name == 'b3':
            net = models.efficientnet_b3(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=1536, out_features=embeding_features)
            return net
        elif name == 'b4':
            net = models.efficientnet_b4(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=1792, out_features=embeding_features)
            return net
        elif name == 'b5':
            net = models.efficientnet_b5(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=2048, out_features=embeding_features)
            return net
        elif name == 'b6':
            net = models.efficientnet_b6(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=2304, out_features=embeding_features)
            return net
        elif name == 'b7':
            net = models.efficientnet_b7(pretrained=pretrained)
            net.classifier[1] = nn.Linear(in_features=2560, out_features=embeding_features)
            return net