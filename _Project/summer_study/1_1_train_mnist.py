import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import sys
sys.path.append("/home/young/chanyoung/Trainer")

from model.lenet import *

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    ])

trainset = torchvision.datasets.MNIST(root='/home/young/chanyoung/Trainer/_DATA', train=True,   
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

model = LeNet_style1()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    loss_list = []
    for images, labels in tqdm(trainloader, 0):
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

        loss_list.append(loss.item())
    
    print(f"Train loss: [{sum(loss_list)/len(loss_list)}]")
print('Finished Training')