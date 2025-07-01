# import pytest
# from src.data_loader import load_data
# from src.preprocess import preprocess_data
# from src.model import train_and_save_best_model

# def test_data_loader():
#     df = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#     assert not df.empty

# def test_preprocess():
#     raw = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#     df = preprocess_data(raw)
#     assert 'Churn' in df.columns

# def test_model_pipeline(tmp_path):
#     raw = load_data('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#     df = preprocess_data(raw)
#     model_path = tmp_path / 'model.pkl'
#     feat_path = tmp_path / 'feat.pkl'
#     X_train, X_test, y_train, y_test = train_and_save_best_model(df, str(model_path), str(feat_path))
#     assert model_path.exists()

import pytest
import os
import joblib
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for tests
from sklearn.datasets import make_classification
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_and_save_best_model
from src.evaluate import evaluate_model

RAW_PATH = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# 1. Data Loader Tests

def test_load_data_returns_dataframe():
    df = load_data(RAW_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_bad_path():
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent.csv')

# 2. Preprocess Tests

def test_preprocess_keeps_churn_and_drops_na_totalcharges():
    raw = load_data(RAW_PATH)
    raw.loc[0, 'TotalCharges'] = 'abc'
    df = preprocess_data(raw)
    assert 'Churn' in df.columns
    assert pd.api.types.is_numeric_dtype(df['TotalCharges'])
    assert df['Churn'].notna().all()


def test_preprocess_missing_churn_raises():
    df = pd.DataFrame({'tenure': [1, 2, 3]})
    with pytest.raises(KeyError):
        preprocess_data(df)

# 3. Model Pipeline Integration Tests

@pytest.fixture(scope='module')
def df_clean():
    raw = load_data(RAW_PATH)
    return preprocess_data(raw)

@pytest.fixture()
def temp_paths(tmp_path):
    m = tmp_path / 'model.pkl'
    f = tmp_path / 'features.pkl'
    return str(m), str(f)


def test_train_and_save_creates_files_and_splits(df_clean, temp_paths):
    model_path, feat_path = temp_paths
    X_train, X_test, y_train, y_test = train_and_save_best_model(df_clean, model_path, feat_path)
    assert os.path.exists(model_path)
    assert os.path.exists(feat_path)
    assert len(X_train) > 0 and len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_saved_model_predict_proba(df_clean, temp_paths):
    model_path, feat_path = temp_paths
    train_and_save_best_model(df_clean, model_path, feat_path)
    features = joblib.load(feat_path)
    model_obj = joblib.load(model_path)
    if isinstance(model_obj, dict):
        model_obj = model_obj['model']
    sample = pd.DataFrame([df_clean.drop(columns=['Churn']).iloc[0]], columns=features)
    proba = model_obj.predict_proba(sample)
    assert proba.shape == (1, 2)

# 4. Synthetic Data Unit Test for Perfect Separation

def test_pipeline_perfect_on_toy(tmp_path):
    X = np.linspace(0, 1, 10)
    df_toy = pd.DataFrame({'f0': X, 'dummy': 0, 'Churn': ['No' if x < 0.5 else 'Yes' for x in X]})
    model_path = tmp_path / 'toy_model.pkl'
    feat_path = tmp_path / 'toy_feat.pkl'
    X_train, X_test, y_train, y_test = train_and_save_best_model(df_toy, str(model_path), str(feat_path))
    metrics = evaluate_model(str(model_path), str(feat_path), X_test, y_test)
    assert metrics['accuracy'] == 1.0

# 5. Parameterized Size Variation Test

@pytest.mark.parametrize('n_samples,n_features', [(50,5), (100,8), (200,10)])
def test_pipeline_various_sizes(tmp_path, n_samples, n_features):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=3, n_redundant=0, random_state=42)
    cols = [f'f{i}' for i in range(n_features)]
    df_var = pd.DataFrame(X, columns=cols)
    df_var['Churn'] = y
    model_path = tmp_path / 'var_model.pkl'
    feat_path = tmp_path / 'var_feat.pkl'
    X_train, X_test, y_train, y_test = train_and_save_best_model(df_var, str(model_path), str(feat_path))
    metrics = evaluate_model(str(model_path), str(feat_path), X_test, y_test)
    assert metrics['accuracy'] > 0.5

# 6. Evaluate Plot Saving Tests

def test_evaluate_saves_plots(tmp_path, df_clean, temp_paths):
    model_path, feat_path = temp_paths
    # Ensure the figures directory exists without deleting existing files
    fig_dir = os.path.join('reports', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    # Run training and evaluation
    X_train, X_test, y_train, y_test = train_and_save_best_model(df_clean, model_path, feat_path)
    evaluate_model(model_path, feat_path, X_test, y_test)
    # Check that the expected figure files exist
    assert os.path.exists(os.path.join(fig_dir, 'confusion_matrix.png'))
    assert os.path.exists(os.path.join(fig_dir, 'roc_curve.png'))

# 7. Negative Test for Bad Model

def test_evaluate_bad_model(tmp_path, df_clean):
    bad = {'notamodel': True}
    path = tmp_path / 'bad.pkl'
    joblib.dump(bad, path)
    with pytest.raises(Exception):
        evaluate_model(str(path), 'none', df_clean.drop(columns=['Churn']), df_clean['Churn'])


