import time
import datetime
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from customdataset import CustomImageDataset
from VGG_model_2node import VGG
import matplotlib.pyplot as plt
from tqdm import tqdm

now = time.localtime()
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
print("[%04d/%02d/%02d %02d:%02d:%02d] Training_start" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))


hyper_param_epoch = 1
hyper_param_batch = 50
hyper_param_learning_rate = 0.0001
model_number = 2
m_name = ['VGG11', 'VGG13', 'VGG16', 'VGG19']
# !!! include cd and then start training!!!!

def main():
    transforms_train = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                               std=[0.2023, 0.1994, 0.2010]), ])

    train_data_set = CustomImageDataset(data_set_path="/home/prml/PSH/SH_dataset/cifar10/train",
                                        transforms=transforms_train)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = train_data_set.num_classes

    hidden_model = VGG(m_name=m_name[model_number], num_classes=num_classes).to(device)

    torch.save(hidden_model.state_dict(),f'./weights/init_weight_2node_210614_({m_name[model_number]})_cifar10.pth')

    print("making original file finished")
if __name__ == '__main__':
    main()