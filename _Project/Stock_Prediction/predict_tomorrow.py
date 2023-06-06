import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def read_data():
    kospi = fdr.StockListing('KOSPI')
    c1 = kospi['Code'].values

    kosdaq = fdr.StockListing('KOSDAQ')
    c2 = kosdaq['Code'].values
    CODES = np.append(c1, c2)
    
    vol_df = pd.DataFrame(columns=CODES)
    change_df = pd.DataFrame(columns=CODES)

    for code in tqdm(CODES):
        df = fdr.DataReader(code, '2000')
        vol_df[code] = df['Volume']
        change_df[code] = df['Change']
        
    vol_df.to_csv("/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo.csv")
    change_df.to_csv("/home/young/chanyoung/Trainer/_DATA/Stock_data/change.csv")

def train_lstm_2():
    batch_size = 1
    scaler = MinMaxScaler()
    change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change.csv",scaler=None, fill_value=0)
    vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo.csv",scaler=scaler, fill_value=-1)

    X, y = train_preprocess(vol_data, change_data, 5, 1)
    X = np.array(X)
    print(len(X), len(X[0]), len(y), len(y[0]))
            
    X_train, X_test, y_train, y_test  = X[:-2], X[-2:], y[:-2], y[-2:]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=17)
    
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    test_dataset = LSTMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
    
    model = LSTM(input_dim=len(X[0]), hidden_dim= 32, output_length=len(y[0]), num_layers=4, batch_size=batch_size)
    
    save_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/tom.pt"
    log_path = f"/home/young/chanyoung/Trainer/_RESULT/Stock/tom/"
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    eval_dic = {"Revenue":StockMeter(), "Average":StocAverageRevenuekMeter(10)}
    
    trainer = Trainer(model=model, max_epoch=100, early_stop=5,
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    save_path=save_path, log_save_path=log_path,
                    optimizer=optimizer, criterion=criterion, evaluate_dic=eval_dic, scheduler=None)
    
    trainer.train()
    
def predict_tom():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    batch_size = 1
    scaler = MinMaxScaler()
    change_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/change.csv",scaler=None, fill_value=0)
    vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/vlo.csv",scaler=scaler, fill_value=-1)
    
    kospi = fdr.StockListing('KOSPI')
    c1 = kospi['Code'].values

    kosdaq = fdr.StockListing('KOSDAQ')
    c2 = kosdaq['Code'].values
    CODES = np.append(c1, c2) 

    X, y = train_preprocess(vol_data, change_data, 5, 1)
    X = np.array(X)
            
    X_train, X_test, y_train, y_test  = X[:-2], X[-2:], y[:-2], y[-2:]
    
    
    x = [vol_data[-5:].flatten().tolist() + change_data[-5:].flatten().tolist()]
    
    
    model = LSTM(input_dim=len(X[0]), hidden_dim= 32, output_length=len(y[0]), num_layers=4, batch_size=batch_size)
    model.load_state_dict(torch.load("/home/young/chanyoung/Trainer/_RESULT/Stock/tom.pt"))
    
    x = np.array(x)
    x = torch.from_numpy(x).float()
    
    x = x.to(device)
    model = model.to(device)
    model.eval()
    
    output = model(x)
    print(output)
    
    o = output.cpu().detach().numpy()
    arr = o.argsort()
    e = arr[0][-10:]
    print(o[0][e])
    print(CODES[e])