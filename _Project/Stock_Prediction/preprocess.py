import pandas as pd

def read_and_remove_nan(csv_path):
    df = pd.read_csv(csv_path)
    print("total nan:",df.isna().sum().sum())
    return df.fillna(-1)

def read_and_preprocess(csv_path, scaler, fill_value):
    df = pd.read_csv(csv_path)
    print("total nan:",df.isna().sum().sum())
    df =  df.fillna(fill_value)
    if scaler is not None:
        scaler.fit(df.drop(columns=["Date"]).values)
        return scaler.transform(df.drop(columns=["Date"]).values)
    else:
        return df.drop(columns=["Date"]).values


def train_preprocess(vol_data, close_data, n, m):
    X = []
    y = []
    for i in range(len(vol_data)-(n+m)):
        x = vol_data[i:i+n].flatten().tolist() + close_data[i:i+n].flatten().tolist()
        # x = x.flatten()
        # x = np.array(x)
        # x = x.reshape((-1,1))
        X.append(x)
        y.append(close_data[i+m+n])
    
    return X,y


