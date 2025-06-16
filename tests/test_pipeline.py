import pytest
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_and_save_best_model

def test_data_loader():
    df = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    assert not df.empty

def test_preprocess():
    raw = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = preprocess_data(raw)
    assert 'Churn' in df.columns

def test_model_pipeline(tmp_path):
    raw = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = preprocess_data(raw)
    model_path = tmp_path / 'model.pkl'
    feat_path = tmp_path / 'feat.pkl'
    X_train, X_test, y_train, y_test = train_and_save_best_model(df, str(model_path), str(feat_path))
    assert model_path.exists()
