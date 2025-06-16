import pandas as pd

def preprocess_data(df):
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
    df = pd.get_dummies(df, drop_first=True)
    return df
