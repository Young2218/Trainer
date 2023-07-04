import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import sys
sys.path.append("/home/young/chanyoung/Trainer")

from model.lenet import *
from trainer.trainer import Trainer

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)


test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    ])

test_set = torchvision.datasets.CIFAR10(root='/home/young/chanyoung/Trainer/_DATA', train=False,   
                                        download=True, transform=test_transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

model = LeNet_3channel()
model = model.to(device)
model.load_state_dict(torch.load("/home/young/chanyoung/Trainer/lenet_with_cifar10.pt"))


# validation
model.eval()        
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())  
        y_true.extend(labels.cpu().numpy())
    
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"F1 score: {f1_score(y_true, y_pred, average='macro')}")