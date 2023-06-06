import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


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


def train_xgboost():    
    scaler = MinMaxScaler()
    clo_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/close_data_2020.csv",scaler=scaler)
    vol_data = read_and_preprocess(csv_path="/home/young/chanyoung/Trainer/_DATA/Stock_data/volume_data_2020.csv",scaler=scaler)
    X, y = train_preprocess(clo_data, vol_data, 10, 1)
    print(len(X), len(X[0]), len(y), len(y[0]))
    
    X_train, X_val, X_test = X[:600], X[600:700], X[700:]
    y_train, y_val, y_test = y[:600], y[600:700], y[700:]
   
    eval_set = [(X_train, y_train), (X_val, y_val)]
 
    model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0,
                             n_estimators=100, learning_rate=0.05, max_depth=8, gamma=0.01,
                             random_state=17, objective='reg:squarederror')
    print("Let's fit")
    model.fit(X_train, y_train, eval_set=eval_set)
    print("end fit")
    
    y_pred = model.predict(X_test)
    print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')
    model.save_model("/home/young/chanyoung/Trainer/_RESULT/Stock/XGB.json")
    # model_xgb_2.load_model("model.json")
    
