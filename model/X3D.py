import torch
import torch.nn as nn
from torchsummary import summary

class X3D(nn.Module):
    def __init__(self, num_classes, model_size: str, pretrained=True) -> None:
        super(X3D, self).__init__()
        if model_size.lower() == 's':
            model_name = 'x3d_s'
        elif model_size.lower() == 'm':
            model_name = 'x3d_m'
        elif model_size.lower() == 'l':
            model_name = 'x3d_l'
        else:
            print("model size is wrong")
        
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
        self.classifier = nn.Linear(400, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x



