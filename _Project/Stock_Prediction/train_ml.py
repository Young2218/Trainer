import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import sys
import torch
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from model.RNNbased import LSTM
from dataset.lstmDataset import LSTMDataset
from trainer import Trainer

from _Project.Stock_Prediction.preprocess import *
from evaluate.stockMetric import *

def train_RF():
    batch_size = 1
    
    scaler = MinMaxScaler()
    change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change_2000.csv",scaler=None, fill_value=0)
    vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo_2000.csv",scaler=scaler, fill_value=-1)
    
    for n in [1,3,5,10]:
        X, y = train_preprocess(change_data, vol_data, n, 1)
        X = np.array(X)
        # print(X.shape)
        print(len(X), len(X[0]), len(y), len(y[0]))
        
        X_train, X_test, y_train, y_test  = X[:5600], X[5600:], y[:5600], y[5600:]
        
        rf = RandomForestRegressor(n_jobs=-1)
        rf.fit(X_train, y_train)
        
        value = 1
        sum = 0
        cnt = 0
        for x,y in zip(X_test, y_test):
            pred = rf.predict(x)
            sort = pred.sort()
            sum += sum(sort[:10])
            cnt += 10
            value *= (1+sort[0])
            
        print(value, sum/cnt)
        
        continue
        
        
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        
        model = LSTM(input_dim=len(X[0]), hidden_dim= 32, output_length=2590, num_layers=2, batch_size=batch_size)
        
        save_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/LSTM_{n}.pt"
        log_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/LSTM_{n}/"
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        eval_dic = {"Revenue":StockMeter(), "Average":StocAverageRevenuekMeter(10)}
        
        trainer = Trainer(model=model, max_epoch=100, early_stop=10,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                        save_path=save_path, log_save_path=log_path,
                        optimizer=optimizer, criterion=criterion, evaluate_dic=eval_dic, scheduler=None)
        
        trainer.train()
        trainer.inference_with_test_loader()
        print(n)
        print("==========================================================")
        
        del train_dataset, val_dataset, test_dataset ,train_loader, val_loader, test_loader, model

