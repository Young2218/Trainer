

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import sys
sys.path.append("/home/young/chanyoung/Trainer")

from model.lenet import *
from trainer.trainer import Trainer

# device = 'cuda' if torch.cuda.is_available else 'cpu'
device = 'cuda'

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    ])

trainset = torchvision.datasets.MNIST(root='/home/young/chanyoung/Trainer/_DATA', train=True,   
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

trainer = Trainer(model=model, max_epoch=100, early_stop=10,
                  train_loader=trainloader, val_loader=None, test_loader= None,
                  save_path= "/home/young/chanyoung/Trainer/_RESULT/summer_study/lenet+mnist/lenet.py", 
                  log_save_path="/home/young/chanyoung/Trainer/_RESULT/summer_study/lenet+mnist/",
                  optimizer=optimizer, criterion=criterion, evaluate_dic=None,scheduler=None)
trainer.train()