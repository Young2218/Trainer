import os
from sklearn.model_selection import train_test_split
from util.basic_util import seed_everything

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision
from sklearn.preprocessing import LabelEncoder

# from trainer import Trainer
from trainer.trainerAngular import Trainer_ang
from trainer.trainer import Trainer

from _Project.ExtraNode.preprocess import *
from dataset.classificationDataset import ClassificationDataset
from loss.angularLoss import AngularPenaltySMLoss
from model.extraNodeMdoel import *

def train_gachon_polyp_for_report(batch_size, lr):
    seed_everything(17)
    experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNodeReport/polyp"
    # csv_path = "/home/prml/chanyoung/Trainer/_Project/ExtraNodeReport/kvasir.csv"
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
   
    train_df = pd.read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/gachon_polyp_train.csv")
    val_df = pd.read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/gachon_polyp_validation.csv")
    test_df = pd.read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/gachon_polyp_test.csv")
    
    le = LabelEncoder()
    le.fit(train_df["class"].values)
    
    train_df["class"] = le.transform(train_df["class"].values)
    # train_df = train_df.sample(100)
    val_df["class"] = le.transform(val_df["class"].values)
    test_df["class"] = le.transform(test_df["class"].values)
    
    
    # df = read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/kvasir.csv")
    # df, encoder = preprocess_kvasir(df, "/home/prml/chanyoung/Trainer/_DATA/kvasir-dataset/")
    
    # print(len(df))
    
    # train_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=17)
    # train_df, val_df, _, _ = train_test_split(train_df, train_df['class'].values, test_size=0.2, random_state=17)
    
    train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
    val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    num_class = len(train_df["class"].unique())
    classes_name = train_df["class"].unique()
    
    #===============================================================================================
    models_info = {'normal':EfficientNet_Normal(num_class=num_class),
                   'arc':EfficientNet_Arc(num_class=num_class),
                   'extra_node':EfficientNet_ExtraNode(num_class=num_class),
                   'arc_extra':EfficientNet_ArcExtraNode(num_class=num_class)}
    
    print(num_class)
    i = 0
    for name, model in models_info.items():
        print(name, "=========================================================")
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

        if i > 1:
            eval_dict = {'accuracy':MulticlassAccuracy(num_classes=num_class+1).to(device)}
        else:
            eval_dict = {'accuracy':MulticlassAccuracy(num_classes=num_class).to(device)}
        i += 1
        
        model_save_path = os.path.join(experiment_root, f"eff_{name}" +".pt")
        log_save_path = os.path.join(experiment_root, f"eff_{name}")
        
        trainer = Trainer_ang(model=model,max_epoch=1000, early_stop=30,
                            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                            save_path=model_save_path, log_save_path=log_save_path,
                            optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler,
                            model_name=name, classes_name=classes_name)
        trainer.train()
        trainer.inference_with_test_loader()

    
def train_gachon_covid_xray_for_report(batch_size, lr):
    seed_everything(17)
    experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNodeReport/xray"
    # csv_path = "/home/prml/chanyoung/Trainer/_Project/ExtraNodeReport/kvasir.csv"
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
   
    train_df = pd.read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/gachon_covid_xray_train.csv")
    val_df = pd.read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/gachon_covid_xray_val.csv")
    test_df = pd.read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/gachon_covid_xray_test.csv")
    
    le = LabelEncoder()
    le.fit(train_df["class"].values)
    
    train_df["class"] = le.transform(train_df["class"].values)
    val_df["class"] = le.transform(val_df["class"].values)
    test_df["class"] = le.transform(test_df["class"].values)
    
    train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
    val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    num_class = len(train_df["class"].unique())
    print(num_class, '============================================')
    classes_name = train_df["class"].unique()
    
    #===============================================================================================
    models_info = {'normal':EfficientNet_Normal(num_class=num_class),
                   'arc':EfficientNet_Arc(num_class=num_class),
                   'extra_node':EfficientNet_ExtraNode(num_class=num_class),
                   'arc_extra':EfficientNet_ArcExtraNode(num_class=num_class)}
    
    print(num_class)
    i = 0
    for name, model in models_info.items():
        print(name, "=========================================================")
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

        if i > 1:
            eval_dict = {'accuracy':MulticlassAccuracy(num_classes=num_class+1).to(device)}
        else:
            eval_dict = {'accuracy':MulticlassAccuracy(num_classes=num_class).to(device)}
        i += 1
        
        model_save_path = os.path.join(experiment_root, f"eff__{name}" +".pt")
        log_save_path = os.path.join(experiment_root, f"eff_{name}")
        
        trainer = Trainer_ang(model=model,max_epoch=1000, early_stop=30,
                            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                            save_path=model_save_path, log_save_path=log_save_path,
                            optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler,
                            model_name=name, classes_name=classes_name)
        trainer.train()
        trainer.inference_with_test_loader()


  
def train_10_person_for_report(batch_size, lr):
    seed_everything(17)
    experiment_root = "/home/prml/chanyoung/Trainer/_RESULT/ExtraNodeReport/10person"
    # csv_path = "/home/prml/chanyoung/Trainer/_Project/ExtraNodeReport/kvasir.csv"
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    df = read_csv("/home/prml/chanyoung/Trainer/_Project/ExtraNode/10person.csv")
    
    print(len(df))
    
    train_df, test_df, _, _ = train_test_split(df, df['class'].values, test_size=0.1, random_state=17)
    train_df, val_df, _, _ = train_test_split(train_df, train_df['class'].values, test_size=0.2, random_state=17)
   
    
    le = LabelEncoder()
    le.fit(train_df["class"].values)
    
    train_df["class"] = le.transform(train_df["class"].values)
    val_df["class"] = le.transform(val_df["class"].values)
    test_df["class"] = le.transform(test_df["class"].values)
    
    train_dataset = ClassificationDataset(train_df["filepath"].values, train_df["class"].values, 224)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    
    val_dataset = ClassificationDataset(val_df["filepath"].values, val_df["class"].values, 224)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    test_dataset = ClassificationDataset(test_df["filepath"].values, test_df["class"].values, 224)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    num_class = len(train_df["class"].unique())
    print(num_class, '============================================')
    classes_name = train_df["class"].unique()
    
    #===============================================================================================
    models_info = {'normal':EfficientNet_Normal(num_class=num_class),
                   'arc':EfficientNet_Arc(num_class=num_class),
                   'extra_node':EfficientNet_ExtraNode(num_class=num_class),
                   'arc_extra':EfficientNet_ArcExtraNode(num_class=num_class)}
    
    print(num_class)
    i = 0
    for name, model in models_info.items():
        print(name, "=========================================================")
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

        if i > 1:
            eval_dict = {'accuracy':MulticlassAccuracy(num_classes=num_class+1).to(device)}
        else:
            eval_dict = {'accuracy':MulticlassAccuracy(num_classes=num_class).to(device)}
        i += 1
        
        model_save_path = os.path.join(experiment_root, f"eff_{name}" +".pt")
        log_save_path = os.path.join(experiment_root, f"eff_{name}")
        
        trainer = Trainer_ang(model=model,max_epoch=1000, early_stop=30,
                            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                            save_path=model_save_path, log_save_path=log_save_path,
                            optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler,
                            model_name=name, classes_name=classes_name)
        trainer.train()
        trainer.inference_with_test_loader()