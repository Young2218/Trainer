import os
from sklearn.model_selection import train_test_split
from util.basic_util import seed_everything

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision

# from trainer import Trainer
from trainer.trainerAngular import Trainer2
from trainer.trainer import Trainer

from _Project.ExtraNode.preprocess import *
from dataset.classificationDataset import ClassificationDataset
from loss.angularLoss import AngularPenaltySMLoss
from model.efficientNet import EfficientNet, EfficientNet_ARC
from model.extraNodeMdoel import EfficientNet_ExtraNode

# def train_vgg_xray(batch_size = 4, lr = 1e-4):
#     seed_everything(17)
#     experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNode/"
    
    
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
#     df = read_csv("/home/prml/chanyoung/Trainer/_DATA/covid_xray/metadata.csv")
#     df, encoder = preprocess(df, "/home/prml/chanyoung/Trainer/_DATA/covid_xray/images/")
    
#     print(len(df))
    
#     train_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=17)
#     train_df, val_df, _, _ = train_test_split(train_df, train_df['class'].values, test_size=0.2, random_state=17)
    
#     train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
#     train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
#     val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
#     val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
#     test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
#     test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)

#     num_class = len(df["class"].unique())
#     model = VGG(model_name='E', num_classes=num_class)
#     model = model.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

#     model_save_path = os.path.join(experiment_root, "VGG_base" +".pt")
#     log_save_path = os.path.join(experiment_root, "VGG_base/")
#     eval_dict = {'acciracy':MulticlassAccuracy(num_classes=num_class).to(device), "f1": MulticlassF1Score(num_classes=num_class).to(device), 
#                  "recall":MulticlassRecall(num_classes=num_class).to(device), "precision": MulticlassPrecision(num_classes=num_class).to(device)}
    
#     trainer = Trainer(model=model,max_epoch=500, early_stop=20,
#                         train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
#                         save_path=model_save_path, log_save_path=log_save_path,
#                         optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler)
#     trainer.train()
#     trainer.inference_with_test_loader()

# def train_vgg_xray_angular_loss(batch_size = 4, lr = 1e-4):
#     seed_everything(17)
#     experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNode/"
    
    
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
#     df = read_csv("/home/prml/chanyoung/Trainer/_DATA/covid_xray/metadata.csv")
#     df, encoder = preprocess(df, "/home/prml/chanyoung/Trainer/_DATA/covid_xray/images/")
    
#     print(len(df))
    
#     train_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=17)
#     train_df, val_df, _, _ = train_test_split(train_df, train_df['class'].values, test_size=0.2, random_state=17)
    
#     train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
#     train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
#     val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
#     val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
#     test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
#     test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)

#     num_class = len(df["class"].unique())
#     model = VGG_Arcface(model_name='E', num_classes=num_class)
#     model = model.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     criterion = nn.CrossEntropyLoss().to(device)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

#     model_save_path = os.path.join(experiment_root, "VGG_arcface" +".pt")
#     log_save_path = os.path.join(experiment_root, "VGG_arcface/")
#     eval_dict = {'acciracy':MulticlassAccuracy(num_classes=num_class).to(device), "f1": MulticlassF1Score(num_classes=num_class).to(device), 
#                  "recall":MulticlassRecall(num_classes=num_class).to(device), "precision": MulticlassPrecision(num_classes=num_class).to(device)}
    
#     trainer = Trainer2(model=model,max_epoch=500, early_stop=20,
#                         train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
#                         save_path=model_save_path, log_save_path=log_save_path,
#                         optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler)
#     trainer.train()
#     trainer.inference_with_test_loader()
    

def train_eff_kvasir(batch_size = 4, lr = 1e-4):
    seed_everything(17)
    experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNode/"
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    df = read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/kvasir.csv")
    df, encoder = preprocess_kvasir(df, "/home/prml/chanyoung/Trainer/_DATA/kvasir-dataset/")
    
    print(len(df))
    
    train_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=17)
    train_df, val_df, _, _ = train_test_split(train_df, train_df['class'].values, test_size=0.2, random_state=17)
    
    train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
    val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)

    num_class = len(df["class"].unique())
    model = EfficientNet(num_class=num_class)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

    model_save_path = os.path.join(experiment_root, "eff_kvasir_base" +".pt")
    log_save_path = os.path.join(experiment_root, "eff_kvasir_base/")
    eval_dict = {'acciracy':MulticlassAccuracy(num_classes=num_class).to(device), "f1": MulticlassF1Score(num_classes=num_class).to(device), 
                 "recall":MulticlassRecall(num_classes=num_class).to(device), "precision": MulticlassPrecision(num_classes=num_class).to(device)}
    
    trainer = Trainer(model=model,max_epoch=500, early_stop=20,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                        save_path=model_save_path, log_save_path=log_save_path,
                        optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler)
    trainer.train()
    trainer.inference_with_test_loader()

def train_eff_kvasir_arc(batch_size = 4, lr = 1e-4):
    seed_everything(17)
    experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNode/"
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    df = read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/kvasir.csv")
    df, encoder = preprocess_kvasir(df, "/home/prml/chanyoung/Trainer/_DATA/kvasir-dataset/")
    
    print(len(df))
    
    train_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=17)
    train_df, val_df, _, _ = train_test_split(train_df, train_df['class'].values, test_size=0.2, random_state=17)
    
    train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
    val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)

    num_class = len(df["class"].unique())
    model = EfficientNet_ARC(num_class=num_class)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

    model_save_path = os.path.join(experiment_root, "eff_kvasir_arcface" +".pt")
    log_save_path = os.path.join(experiment_root, "eff_kvasir__arcface/")
    eval_dict = {'acciracy':MulticlassAccuracy(num_classes=num_class).to(device), "f1": MulticlassF1Score(num_classes=num_class).to(device), 
                 "recall":MulticlassRecall(num_classes=num_class).to(device), "precision": MulticlassPrecision(num_classes=num_class).to(device)}
    
    trainer = Trainer2(model=model,max_epoch=500, early_stop=20,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                        save_path=model_save_path, log_save_path=log_save_path,
                        optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler)
    trainer.train()
    trainer.inference_with_test_loader()

def train_eff_extra_kvasir_arc(batch_size = 4, lr = 1e-4):
    seed_everything(17)
    experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNode/"
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    df = read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/kvasir.csv")
    df, encoder = preprocess_kvasir(df, "/home/prml/chanyoung/Trainer/_DATA/kvasir-dataset/")
    
    print(len(df))
    
    train_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=17)
    train_df, val_df, _, _ = train_test_split(train_df, train_df['class'].values, test_size=0.2, random_state=17)
    
    train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
    val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)

    num_class = len(df["class"].unique())
    model = EfficientNet_ExtraNode(num_class=num_class)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

    model_save_path = os.path.join(experiment_root, "eff_kvasir_extra_arcface" +".pt")
    log_save_path = os.path.join(experiment_root, "eff_kvasir_extra_arcface/")
    eval_dict = {'acciracy':MulticlassAccuracy(num_classes=num_class+1).to(device), "f1": MulticlassF1Score(num_classes=num_class+1).to(device), 
                 "recall":MulticlassRecall(num_classes=num_class+1).to(device), "precision": MulticlassPrecision(num_classes=num_class+1).to(device)}
    
    trainer = Trainer2(model=model,max_epoch=500, early_stop=20,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                        save_path=model_save_path, log_save_path=log_save_path,
                        optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler)
    trainer.train()
    trainer.inference_with_test_loader()