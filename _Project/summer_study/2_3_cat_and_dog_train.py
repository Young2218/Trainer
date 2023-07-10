import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

from tqdm import tqdm

import sys
sys.path.append("/home/young/chanyoung/Trainer")

from model.vgg import VGG
from _Project.summer_study.cat_dog_dataset import *



device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)



train_df = pd.read_csv("/home/young/chanyoung/Trainer/_Project/summer_study/2_cat_and_dog_train.csv")
test_df = pd.read_csv("/home/young/chanyoung/Trainer/_Project/summer_study/2_cat_and_dog_test.csv")

le = LabelEncoder()
le.fit(train_df['label'])
train_df['label'] = le.transform(train_df['label'])
test_df['label'] = le.transform(test_df['label'])

train_dataset = CustomDataset(img_paths=train_df["path"].values,
                              labels=train_df['label'].values,
                              resize=224,
                              transforms=None)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

test_set = CustomDataset(img_paths=test_df["path"].values,
                         labels=test_df['label'].values,
                         resize=224,
                         transforms=None)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

model = torchvision.models.vgg11_bn(weights='VGG11_BN_Weights.IMAGENET1K_V1')

# 1 modify last layer
# model.classifier[6] = nn.Linear(4096, 2) # number of classes, cat and dog
# print(model)

# 2 add new layer
model.classifier.append(nn.Linear(1000, 2)) # number of classes, cat and dog
print(model)
# model = VGG(model_name='E',num_classes=2)


model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

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

# ================================================================
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