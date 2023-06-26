import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_csv(path):
    return pd.read_csv(path)

def preprocess(df, prefix_path):
    # drop unnecessary columns
    df = df[["finding", "filename"]]
    
    # label encoding
    encoder = LabelEncoder()
    df["class"] = encoder.fit_transform(df["finding"])
    df = df[~df["filename"].str.contains("nii.gz")]
    df["filepath"] = prefix_path + df["filename"]
    
    return df, encoder

def preprocess_kvasir(df, prefix_path):
    # drop unnecessary columns
    df = df[["class", "filename"]]
    
    # label encoding
    encoder = LabelEncoder()
    df["class"] = encoder.fit_transform(df["class"])
    df = df[~df["filename"].str.contains("nii.gz")]
    df["filepath"] = prefix_path + df["filename"]
    
    return df, encoder

if __name__ == "__main__": 
    df = read_csv("/home/prml/chanyoung/Trainer/_DATA/covid_xray/metadata.csv")
    df, encoder = preprocess(df, "/home/prml/chanyoung/Trainer/_DATA/covid_xray/images/")
    print(df.info())
    
    import cv2
    print(df.iloc[0]["filepath"])
    
    img = cv2.imread(df.iloc[0]["filepath"])
    print(img.shape)
    cv2.imshow( "hello", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(df.head())