from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('deployment/model.pkl')
feature_cols = joblib.load('deployment/feature_cols.pkl')

class CustomerData(BaseModel):
    data: dict

app = FastAPI(title='Telco Churn API')

@app.post('/predict')
def predict(request: CustomerData):
    df = pd.DataFrame([request.data])
    df_enc = pd.get_dummies(df)
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_cols]
    try:
        pred = model.predict(df_enc)[0]
        prob = model.predict_proba(df_enc)[0,1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {'churn_prediction': int(pred), 'churn_probability': float(prob)}

@app.get('/health')
def health():
    return {'status':'ok'}
