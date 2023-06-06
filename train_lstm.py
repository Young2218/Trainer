import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
# from torch.utils.data import DataLoader
import sys

# sys.path.append('/home/prml/workspace/Trainer')
from model.basicLSTM import BasicLSTM
from dataset.lstmDataset import LSTMDataset


def read_and_remove_nan(csv_path):
    df = pd.read_csv(csv_path)
    print("total nan:",df.isna().sum().sum())
    return df.fillna(-1)

def read_and_preprocess(csv_path, scaler):
    df = pd.read_csv(csv_path)
    print("total nan:",df.isna().sum().sum())
    df =  df.fillna(-1)
    scaler.fit(df.drop(columns=["Date"]).values)
    return scaler.transform(df.drop(columns=["Date"]).values)

def train_preprocess(vol_data, close_data, n, m):
    X = []
    y = []
    for i in range(len(vol_data)-(n+m)):
        x = vol_data[i:i+n].flatten().tolist() + close_data[i:i+n].flatten().tolist()
        # x = x.flatten()
        X.append(x)
        y.append(close_data[i+m])
    
    return X,y

if __name__ == "__main__":
    
    batch_size = 4
    
    # 데이터 읽고 트레인 데이터까지 생성
    scaler = MinMaxScaler()
    clo_data = read_and_preprocess(csv_path="/home/prml/workspace/23_ml_term_project/close_data_2020.csv",scaler=scaler)
    vol_data = read_and_preprocess(csv_path="/home/prml/workspace/23_ml_term_project/volume_data_2020.csv",scaler=scaler)
    X, y = train_preprocess(clo_data, vol_data, 10, 1)
    print(len(X), len(X[0]), len(y), len(y[0]))
    
    X_train, X_val, X_test = X[:600], X[600:700], X[700:]
    y_train, y_val, y_test = y[:600], y[600:700], y[700:]
    
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    test_dataset = LSTMDataset(X_test, y_test)
    
    # train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)
    
    model = BasicLSTM(54340, 5, 2717)
    
    # model.train()
    # for idx, batch in enumerate(train_iter):
    #     prediction = model(text)
    #     loss = loss_fn(prediction, target)
    #     wandb.log({"Training Loss": loss.item()})
    #     num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
    #     acc = 100.0 * num_corrects/len(batch)
    #     wandb.log({"Training Accuracy": acc.item()})
