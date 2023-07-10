import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import sys
sys.path.append("/home/young/chanyoung/Trainer")

from model.lenet import *
from model.alexnet import AlexNet
from trainer.trainer import Trainer

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)

train_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    ])

test_transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    ])

train_dataset = torchvision.datasets.CIFAR10(root='/home/young/chanyoung/Trainer/_DATA', train=True,   
                                        download=True, transform=train_transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

test_set = torchvision.datasets.CIFAR10(root='/home/young/chanyoung/Trainer/_DATA', train=False,   
                                        download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

model = AlexNet(1000)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

min_val_loss = float('inf')
for epoch in range(10):  # loop over the dataset multiple times
    # training
    model.train()
    train_loss_list = []
    for images, labels in tqdm(train_loader, 0):
        # get the inputs
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())
    train_loss = sum(train_loss_list)/len(train_loss_list)
    
    # validation
    model.eval()        
    val_loss_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_list.append(loss)
    
    val_loss = sum(val_loss_list)/len(val_loss_list)
    
    print(f"train loss: [{train_loss}] val loss: [{val_loss}]")
    
    if min_val_loss > val_loss:
        torch.save(model.state_dict(), "lenet_with_cifar10.pt")
    

print('Finished Training')