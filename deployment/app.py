from typing import Literal
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Load model & features ────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "deployment/model.pkl")
FEATS_PATH = os.getenv("FEATS_PATH", "deployment/feature_cols.pkl")

model_data = joblib.load(MODEL_PATH)
if isinstance(model_data, dict) and "model" in model_data and "threshold" in model_data:
    model = model_data["model"]
    saved_threshold = float(model_data["threshold"])
else:
    model = model_data
    saved_threshold = None

feature_cols = joblib.load(FEATS_PATH) if os.path.exists(FEATS_PATH) else None


# ─── Pydantic schemas ────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    # Enumerations make docs clearer & validate inputs
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1, description="0 or 1")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, description="Months of tenure")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    class Config:  # Pydantic v1
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.7,
                "TotalCharges": 430.5,
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: int = Field(description="Predicted label: 1=churn, 0=no churn")
    churn_probability: float = Field(ge=0, le=1, description="Churn probability (0–1, rounded)")
    threshold: float = Field(ge=0, le=1, description="Decision threshold used")


# ─── FastAPI app setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Telecom Churn API",
    version="1.0",
    docs_url=None,   # keep custom docs
    redoc_url=None,
)

# CORS (relax for demo; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve custom docs UI (expects /openapi.json)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    return FileResponse("static/docs.html")


# ─── Root / Home – simple landing with button to docs ────────────────────────
@app.get("/", response_class=HTMLResponse, summary="Welcome / Health Check")
def root_and_health():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Telecom Churn API</title>
        <style>
          body { display:flex; flex-direction:column; justify-content:center; align-items:center;
                 height:100vh; margin:0; background:linear-gradient(135deg,#4b6cb7 0%,#182848 100%);
                 color:#fff; font-family:Arial,sans-serif; text-align:center; }
          h1 { font-size:2em; margin-bottom:.5em; }
          .btn { background:#ffcc00; color:#182848; border:none; padding:1em 2em; font-size:1.2em;
                 border-radius:6px; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,.3); }
          .btn:hover { background:#ffd633; }
        </style>
      </head>
      <body>
        <h1>Welcome to Telecom Churn Prediction</h1>
        <button class="btn" onclick="window.location.href='/docs'">Try Predicting AI</button>
      </body>
    </html>
    """


# ─── Utilities ────────────────────────────────────────────────────────────────
def _align_columns(df_enc: pd.DataFrame) -> pd.DataFrame:
    """Align encoded columns to training order; add missing with 0, drop extras, reorder."""
    if feature_cols is None:
        return df_enc
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc.reindex(columns=feature_cols, fill_value=0)
    return df_enc


def _predict_from_df(df: pd.DataFrame) -> PredictionResponse:
    # One-hot encode and align features
    df_enc = pd.get_dummies(df)
    df_enc = _align_columns(df_enc)

    # Probability
    proba: float
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df_enc)[:, 1][0])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(df_enc)[0])
        proba = float(1.0 / (1.0 + np.exp(-score)))  # logistic mapping
    else:
        # fallback—not ideal, but avoids crashing
        pred = int(model.predict(df_enc)[0])
        proba = float(pred)

    # Decision threshold
    threshold_used = saved_threshold if saved_threshold is not None else 0.5
    pred_label = int(proba >= threshold_used)

    return PredictionResponse(
        churn_prediction=pred_label,
        churn_probability=round(proba, 2),
        threshold=threshold_used,
    )


# ─── Health ──────────────────────────────────────────────────────────────────
@app.get("/healthz", tags=["Meta"])
def healthz():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_features": 0 if feature_cols is None else len(feature_cols),
        "threshold": saved_threshold if saved_threshold is not None else 0.5,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post(
    "/predict/json",
    summary="Predict churn (JSON body)",
    response_model=PredictionResponse,
)
def predict_json(payload: CustomerData = Body(..., description="Customer attributes")):
    """
    Accept a JSON body conforming to CustomerData; returns prediction & probability.
    """
    df = pd.DataFrame([payload.dict()])  # pydantic v1
    return _predict_from_df(df)


@app.get(
    "/predict/query",
    summary="Predict churn (query params)",
    response_model=PredictionResponse,
)
def predict_query(
    gender: Literal["Male", "Female"] = Query(..., example="Female"),
    SeniorCitizen: int = Query(..., ge=0, le=1, example=0),
    Partner: Literal["Yes", "No"] = Query(..., example="No"),
    Dependents: Literal["Yes", "No"] = Query(..., example="No"),
    tenure: int = Query(..., ge=0, example=5),
    PhoneService: Literal["Yes", "No"] = Query(..., example="Yes"),
    MultipleLines: Literal["Yes", "No", "No phone service"] = Query(..., example="No"),
    InternetService: Literal["DSL", "Fiber optic", "No"] = Query(..., example="Fiber optic"),
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Query(..., example="No"),
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Query(..., example="No"),
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Query(..., example="No"),
    TechSupport: Literal["Yes", "No", "No internet service"] = Query(..., example="No"),
    StreamingTV: Literal["Yes", "No", "No internet service"] = Query(..., example="Yes"),
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Query(..., example="Yes"),
    Contract: Literal["Month-to-month", "One year", "Two year"] = Query(..., example="Month-to-month"),
    PaperlessBilling: Literal["Yes", "No"] = Query(..., example="Yes"),
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Query(..., example="Electronic check"),
    MonthlyCharges: float = Query(..., ge=0, example=85.7),
    TotalCharges: float = Query(..., ge=0, example=430.5),
):
    payload = CustomerData(
        gender=gender,
        SeniorCitizen=SeniorCitizen,
        Partner=Partner,
        Dependents=Dependents,
        tenure=tenure,
        PhoneService=PhoneService,
        MultipleLines=MultipleLines,
        InternetService=InternetService,
        OnlineSecurity=OnlineSecurity,
        OnlineBackup=OnlineBackup,
        DeviceProtection=DeviceProtection,
        TechSupport=TechSupport,
        StreamingTV=StreamingTV,
        StreamingMovies=StreamingMovies,
        Contract=Contract,
        PaperlessBilling=PaperlessBilling,
        PaymentMethod=PaymentMethod,
        MonthlyCharges=MonthlyCharges,
        TotalCharges=TotalCharges,
    )
    df = pd.DataFrame([payload.dict()])
    return _predict_from_df(df)
