import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # 데이터 읽고 트레인 데이터까지 생성
    scaler = MinMaxScaler()
    clo_data = read_and_preprocess(csv_path="/home/prml/workspace/23_ml_term_project/close_data_2020.csv",scaler=scaler)
    vol_data = read_and_preprocess(csv_path="/home/prml/workspace/23_ml_term_project/volume_data_2020.csv",scaler=scaler)
    X, y = train_preprocess(clo_data, vol_data, 10, 1)
    print(len(X), len(X[0]), len(y), len(y[0]))
    
    
    