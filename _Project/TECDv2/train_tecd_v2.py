from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A


from model.tecdv2 import TECDv2
from dataset.segmentationDataset import SegmentationDataset
from trainer import Trainer
from util.basic_util import seed_everything
from loss.diceLoss import DiceLoss
from loss.focalLoss import FocalLoss
from evaluate.segmentationMetric import *



def train_TECDv2():
    seed = 17
    img_size = 512
    batch_size = 4
    learning_rate = 1e-4
    experiment_root = "/home/young/chanyoung/Trainer/_RESULT/polyp_segmentation/"
    
    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
    model = TECDv2()
    
    seed_everything(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
    df = pd.read_csv("/home/young/chanyoung/Experiment/polyp_segmentation/csv/kvasir_seg.csv")
    # df = df.sample(20)
    
    my_transform = A.Compose([
                                A.Resize(img_size, img_size),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.Rotate(180,p=1)
                            ])

    train_df, test_df, _, _ = train_test_split(df, df['mask'].values, test_size=0.1, random_state=seed)
    train_df, val_df, _, _ = train_test_split(train_df, train_df['mask'].values, test_size=0.2, random_state=seed)
    
    train_dataset = SegmentationDataset(train_df['image'].values, train_df['mask'].values, img_size=img_size, 
                                        transforms=my_transform, feature_extractor=feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)

    val_dataset = SegmentationDataset(val_df['image'].values, val_df['mask'].values, img_size=img_size,
                                      transforms=None, feature_extractor=feature_extractor)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    test_dataset = SegmentationDataset(test_df['image'].values, test_df['mask'].values, img_size=img_size,
                                      transforms=None, feature_extractor=feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    
    # /////////////////////////////////////////////////////////////////////////////
    
    
    model = model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = DiceLoss().to(device)
    criterion = FocalLoss(alpha=0.25).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, factor=0.5)

    model_save_path = os.path.join(experiment_root, "TECDv2_focal" +".pt")
    log_save_path = os.path.join(experiment_root, "TECDv2_focal/")
    eval_dict = {'DICE':DiceCoef(), "mIOU":meanIOU()}
    
    
    # model = nn.DataParallel(model)
    #     #/////////////////////// SETTING SOLVER ////////////////////////////////////// 
    trainer = Trainer(model=model,max_epoch=500, early_stop=20,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                        save_path=model_save_path, log_save_path=log_save_path,
                        optimizer=optimizer, criterion=criterion, evaluate_dic= eval_dict, scheduler=scheduler)
    trainer.train()
    trainer.inference_with_test_loader()
        


