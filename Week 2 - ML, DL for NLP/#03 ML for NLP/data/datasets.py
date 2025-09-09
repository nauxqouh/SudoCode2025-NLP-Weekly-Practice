import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = ['label', 'text']
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )
    return X_train, X_test, y_train, y_test
