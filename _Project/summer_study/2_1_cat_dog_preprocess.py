import os
import pandas as pd

root_train = "/home/young/chanyoung/Trainer/_DATA/CATandDOG/training_set/training_set"

label_list = []
path_list = []

for label_name in os.listdir(root_train):
    label_path = os.path.join(root_train, label_name)
    
    for img_name in os.listdir(label_path):
        if '.jpg' not in img_name:
            continue
        img_path = os.path.join(label_path, img_name)
        label_list.append(label_name)
        path_list.append(img_path)

train_df = pd.DataFrame({'label':label_list, 'path':path_list})
train_df.to_csv('/home/young/chanyoung/Trainer/_Project/summer_study/2_cat_and_dog_train.csv', index=False)

root_test = "/home/young/chanyoung/Trainer/_DATA/CATandDOG/test_set/test_set"

label_list = []
path_list = []

for label_name in os.listdir(root_train):
    label_path = os.path.join(root_train, label_name)
    
    for img_name in os.listdir(label_path):
        if '.jpg' not in img_name:
            continue
        img_path = os.path.join(label_path, img_name)
        label_list.append(label_name)
        path_list.append(img_path)

test_df = pd.DataFrame({'label':label_list, 'path':path_list})
test_df.to_csv('/home/young/chanyoung/Trainer/_Project/summer_study/2_cat_and_dog_test.csv', index=False)