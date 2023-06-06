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

from model.RNNbased import LSTM, RNN, GRU
from dataset.lstmDataset import LSTMDataset
from trainer import Trainer

from _Project.Stock_Prediction.preprocess import *
from evaluate.stockMetric import *

def train_lstm():
    batch_size = 1
    for n in [3]:
        scaler = MinMaxScaler()
        change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change_2000.csv",scaler=None, fill_value=0)
        vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo_2000.csv",scaler=scaler, fill_value=-1)
    
        X, y = train_preprocess(vol_data, change_data, n, 1)
        X = np.array(X)
        # print(X.shape)
        print(len(X), len(X[0]), len(y), len(y[0]))
                
        X_train, X_test, y_train, y_test  = X[:5600], X[5600:], y[:5600], y[5600:]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=17)
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        
        model = LSTM(input_dim=len(X[0]), hidden_dim= 500, output_length=len(y[0]), num_layers=50, batch_size=batch_size)
        
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

def train_rnn():
    batch_size = 1
    for n in [1,3,5]:
        scaler = MinMaxScaler()
        change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change_2000.csv",scaler=None, fill_value=0)
        vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo_2000.csv",scaler=scaler, fill_value=-1)
    
        X, y = train_preprocess(vol_data, change_data, n, 1)
        X = np.array(X)
        # print(X.shape)
        print(len(X), len(X[0]), len(y), len(y[0]))
                
        X_train, X_test, y_train, y_test  = X[:5600], X[5600:], y[:5600], y[5600:]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=17)
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        
        model = RNN(input_dim=len(X[0]), hidden_dim= 32, output_length=len(y[0]), num_layers=2, batch_size=batch_size)
        
        save_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/RNN_{n}.pt"
        log_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/RNN_{n}/"
        
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

def train_gru():
    batch_size = 1
    for n in [10]:
        scaler = MinMaxScaler()
        change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change_2000.csv",scaler=None, fill_value=0)
        vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo_2000.csv",scaler=scaler, fill_value=-1)
    
        X, y = train_preprocess(vol_data, change_data, n, 1)
        X = np.array(X)
        # print(X.shape)
        print(len(X), len(X[0]), len(y), len(y[0]))
                
        X_train, X_test, y_train, y_test  = X[:5600], X[5600:], y[:5600], y[5600:]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=17)
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        
        model = GRU(input_dim=len(X[0]), hidden_dim= 32, output_length=2590, num_layers=2, batch_size=batch_size)
        
        save_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/GRU_{n}.pt"
        log_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/GRU_{n}/"
        
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
        
def fine_tune_rnn():
    batch_size = 1
    for n in [10]:
        scaler = MinMaxScaler()
        change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change_2000.csv",scaler=None, fill_value=0)
        vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo_2000.csv",scaler=scaler, fill_value=-1)
    
        X, y = train_preprocess(change_data, vol_data, n, 1)
        X = np.array(X)
        # print(X.shape)
        print(len(X), len(X[0]), len(y), len(y[0]))
                
        X_train, X_test, y_train, y_test  = X[:5600], X[5600:], y[:5600], y[5600:]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=17)
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        
        model = RNN(input_dim=len(X[0]), hidden_dim= 64, output_length=2590, num_layers=4, batch_size=batch_size)
        
        save_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/RNN_{n}.pt"
        log_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/RNN_{n}/"
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = torch.nn.MSELoss()
        eval_dic = {"Revenue":StockMeter(), "Average":StocAverageRevenuekMeter(10)}
        
        trainer = Trainer(model=model, max_epoch=100, early_stop=5,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                        save_path=save_path, log_save_path=log_path,
                        optimizer=optimizer, criterion=criterion, evaluate_dic=eval_dic, scheduler=None)
        
        trainer.train()
        trainer.inference_with_test_loader()
        print(n)
        print("==========================================================")
        
        del train_dataset, val_dataset, test_dataset ,train_loader, val_loader, test_loader, model

def fine_tune_lstm():
    batch_size = 1
    for n in [2]:
        scaler = MinMaxScaler()
        change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change_2000.csv",scaler=None, fill_value=0)
        vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo_2000.csv",scaler=scaler, fill_value=-1)
    
        X, y = train_preprocess(change_data, vol_data, n, 1)
        X = np.array(X)
        # print(X.shape)
        print(len(X), len(X[0]), len(y), len(y[0]))
                
        X_train, X_test, y_train, y_test  = X[:5600], X[5600:], y[:5600], y[5600:]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=17)
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        
        model = LSTM(input_dim=len(X[0]), hidden_dim= 128, output_length=2590, num_layers=8, batch_size=batch_size)
        
        save_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/fine_LSTM2_{n}.pt"
        log_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/fine_LSTM2_{n}/"
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        eval_dic = {"Revenue":StockMeter(), "Average":StocAverageRevenuekMeter(10)}
        
        trainer = Trainer(model=model, max_epoch=100, early_stop=5,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                        save_path=save_path, log_save_path=log_path,
                        optimizer=optimizer, criterion=criterion, evaluate_dic=eval_dic, scheduler=None)
        
        trainer.train()
        trainer.inference_with_test_loader()
        print(n)
        print("==========================================================")
        
        del train_dataset, val_dataset, test_dataset ,train_loader, val_loader, test_loader, model

def fine_tune_gru():
    batch_size = 1
    for n in [3]:
        scaler = MinMaxScaler()
        change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change_2000.csv",scaler=None, fill_value=0)
        vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo_2000.csv",scaler=scaler, fill_value=-1)
    
        X, y = train_preprocess(change_data, vol_data, n, 1)
        X = np.array(X)
        # print(X.shape)
        print(len(X), len(X[0]), len(y), len(y[0]))
                
        X_train, X_test, y_train, y_test  = X[:5600], X[5600:], y[:5600], y[5600:]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=17)
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val)
        test_dataset = LSTMDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
        
        model = GRU(input_dim=len(X[0]), hidden_dim= 64, output_length=2590, num_layers=4, batch_size=batch_size)
        
        save_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/fine_GRU_{n}.pt"
        log_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/fine_GRU_{n}/"
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
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