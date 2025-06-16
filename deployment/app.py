from fastapi import FastAPI, HTTPException, Query,Body
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# Load trained model + feature list
model = joblib.load('deployment/model.pkl')
feature_cols = joblib.load('deployment/feature_cols.pkl')

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI(
    title="Telecom Churn API",
    version="1.0",
    docs_url=None,
    redoc_url=None           
    # docs_url="/docs",
    # redoc_url="/redoc"  
    # redoc_url="/docs"      
)
### to get clean Ui
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    return FileResponse("static/docs.html")

@app.get("/", summary="Welcome/Health Check")
def root_and_health():
    return {
        "message": "Welcome to the Telco Churn API. Use /docs for details.",
        "status": "ok"
    }

def _make_prediction(data: pd.DataFrame):
    # One-hot encode and align features
    df_enc = pd.get_dummies(data)
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_cols]

    # Predict
    pred = model.predict(df_enc)[0]
    prob = model.predict_proba(df_enc)[0, 1]
    return {"churn_prediction": int(pred), "churn_probability": float(prob)}

# JSON body endpoint
@app.post(
    "/predict/json",
    summary="Predict churn via JSON body",
    response_model=dict
)
def predict_json(
    payload: CustomerData = Body(
        ...,
        example={
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 350.5
        }
    )
):
    df = pd.DataFrame([payload.dict()])
    return _make_prediction(df)

# Query parameter endpoint
@app.get(
    "/predict/query",
    summary="Predict churn via query parameters",
    response_model=dict
)
def predict_query(
    gender: str                = Query(..., description="Male/Female",example ="Female"),
    SeniorCitizen: int         = Query(..., ge=0, le=1, description="0 or 1",example =1),
    Partner: str               = Query(..., description="Yes/No",example ="Yes"),
    Dependents: str            = Query(..., description="Yes/No",example ="Yes"),
    tenure: int                = Query(..., ge=0, description="Months of tenure",example ="12"),
    PhoneService: str          = Query(..., description="Yes/No",example ="Yes"),
    MultipleLines: str         = Query(..., description="Yes/No/No phone service",example ="Yes"),
    InternetService: str       = Query(..., description="DSL/Fiber optic/No",example ="DSL"),
    OnlineSecurity: str        = Query(..., description="Yes/No/No internet service",example ="Yes"),
    OnlineBackup: str          = Query(..., description="Yes/No/No internet service",example ="Yes"),
    DeviceProtection: str      = Query(..., description="Yes/No/No internet service",example ="Yes"),
    TechSupport: str           = Query(..., description="Yes/No/No internet service",example ="Yes"),
    StreamingTV: str           = Query(..., description="Yes/No/No internet service",example ="Yes"),
    StreamingMovies: str       = Query(..., description="Yes/No/No internet service",example ="Yes"),
    Contract: str              = Query(..., description="Month-to-month/One year/Two year",example ="One year"),
    PaperlessBilling: str      = Query(..., description="Yes/No",example ="Yes"),
    PaymentMethod: str         = Query(..., description="Electronic check/Mailed check/Bank transfer/Credit card",example ="Mailed check"),
    MonthlyCharges: float      = Query(..., gt=0, description="Monthly charges",example =29.85),
    TotalCharges: float        = Query(..., ge=0, description="Total charges",example =390.50)
):
    # Build DataFrame from query params
    data = {k: v for k, v in locals().items() if k in CustomerData.__fields__}
    df = pd.DataFrame([data])
    return _make_prediction(df)