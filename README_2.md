# Telco Customer Churn Prediction Project

## Overview
Predict customer churn using machine learning, with EDA, full testing, and deployment pipelines.

## Structure
- data/raw/: Place your dataset CSV here.
- notebooks/: Jupyter notebooks for exploration.
- src/: Source modules (`data_loader`, `preprocess`, `model`, `evaluate`, `eda`).
- reports/figures/: Automatically saved plots.
- tests/: pytest unit tests.
- deployment/: FastAPI application.
- requirements.txt: Python dependencies.
- main.py: Entry point to run EDA, train, evaluate, and save model.
# 1a) Create venv (only once)
python -m venv venv

# 1b) Activate it
.\venv\Scripts\Activate.ps1

# Commands
-- pip install --upgrade pip
-- pip install -r requirements.txt