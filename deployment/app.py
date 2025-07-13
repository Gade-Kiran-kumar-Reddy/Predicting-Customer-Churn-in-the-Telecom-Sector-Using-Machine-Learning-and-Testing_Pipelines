from fastapi import FastAPI, Query, Body
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

# ─── Load model & features ────────────────────────────────────────────────────
model_data = joblib.load('deployment/model.pkl')
if isinstance(model_data, dict) and 'model' in model_data and 'threshold' in model_data:
    model = model_data['model']
    threshold = model_data['threshold']
else:
    model = model_data
    threshold = None

feature_cols = joblib.load('deployment/feature_cols.pkl')


# ─── Pydantic schema ──────────────────────────────────────────────────────────
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


# ─── FastAPI app setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Telecom Churn API",
    version="1.0",
    docs_url=None,
    redoc_url=None
)

# Serve custom docs UI
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    return FileResponse("static/docs.html")

# ─── Root / Home – HTML page with background, welcome message and button ─────
@app.get("/", response_class=HTMLResponse, summary="Welcome / Health Check")
def root_and_health():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Telecom Churn API</title>
        <style>
          body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
            color: #fff;
            font-family: Arial, sans-serif;
            text-align: center;
          }
          h1 {
            font-size: 2em;
            margin-bottom: 0.5em;
          }
          .btn {
            background: #ffcc00;
            color: #182848;
            border: none;
            padding: 1em 2em;
            font-size: 1.2em;
            border-radius: 4px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            transition: background 0.3s;
          }
          .btn:hover {
            background: #ffd633;
          }
        </style>
      </head>
      <body>
        <h1>Welcome to Telecom Churn Prediction Model 1.0</h1>
        <button class="btn" onclick="window.location.href='/docs'">
          Try it
        </button>
      </body>
    </html>
    """


# ─── Prediction helper ────────────────────────────────────────────────────────
def _make_prediction(df: pd.DataFrame):
    # One-hot encode and align features
    df_enc = pd.get_dummies(df)
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[feature_cols]

    # Predict
    pred = model.predict(df_enc)[0]
    proba = float(model.predict_proba(df_enc)[0, 1])
    proba = round(proba, 2)

    # Apply threshold if calibrated
    if threshold is not None:
        pred = int(proba > threshold)

    return {"churn_prediction": int(pred), "churn_probability": proba}


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post(
    "/predict/json",
    summary="Predict churn (JSON body)",
    response_model=dict,
)
def predict_json(payload: CustomerData = Body(...)):
    """
    Accept a JSON body conforming to CustomerData; returns prediction & 2-dp probability.
    """
    df = pd.DataFrame([payload.dict()])
    return _make_prediction(df)


@app.get(
    "/predict/query",
    summary="Predict churn (query params)",
    response_model=dict,
)
def predict_query(
    gender: str           = Query(..., description="Male/Female", example="Female"),
    SeniorCitizen: int    = Query(..., ge=0, le=1, description="0 or 1", example=0),
    Partner: str          = Query(..., description="Yes/No", example="Yes"),
    Dependents: str       = Query(..., description="Yes/No", example="No"),
    tenure: int           = Query(..., ge=0, description="Months of tenure", example=12),
    PhoneService: str     = Query(..., description="Yes/No", example="Yes"),
    MultipleLines: str    = Query(..., description="Yes/No/No phone service", example="No"),
    InternetService: str  = Query(..., description="DSL/Fiber optic/No", example="DSL"),
    OnlineSecurity: str   = Query(..., description="Yes/No/No internet service", example="No"),
    OnlineBackup: str     = Query(..., description="Yes/No/No internet service", example="Yes"),
    DeviceProtection: str = Query(..., description="Yes/No/No internet service", example="No"),
    TechSupport: str      = Query(..., description="Yes/No/No internet service", example="No"),
    StreamingTV: str      = Query(..., description="Yes/No/No internet service", example="No"),
    StreamingMovies: str  = Query(..., description="Yes/No/No internet service", example="No"),
    Contract: str         = Query(..., description="Month-to-month/One year/Two year", example="Month-to-month"),
    PaperlessBilling: str = Query(..., description="Yes/No", example="Yes"),
    PaymentMethod: str    = Query(..., description="Electronic check/Mailed check/Bank transfer/Credit card", example="Electronic check"),
    MonthlyCharges: float = Query(..., gt=0, description="Monthly charges", example=29.85),
    TotalCharges: float   = Query(..., ge=0, description="Total charges", example=350.5),
):
    data = {k: v for k, v in locals().items() if k in CustomerData.__fields__}
    df = pd.DataFrame([data])
    return _make_prediction(df)


# ─── Override OpenAPI to start JSON box truly blank ───────────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)

    # Remove existing example/examples and inject a blank object
    try:
        rb = schema["paths"]["/predict/json"]["post"]["requestBody"]["content"]["application/json"]
        rb.pop("example", None)
        rb.pop("examples", None)
        rb["example"] = {}
    except KeyError:
        pass

    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi
