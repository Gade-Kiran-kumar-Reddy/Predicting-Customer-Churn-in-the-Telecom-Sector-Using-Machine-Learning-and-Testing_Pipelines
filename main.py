import os
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.eda import run_eda
from src.model import train_and_save_model
from src.evaluate import evaluate_model

def main():
    # Create necessary dirs
    os.makedirs('deployment', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    # Paths
    raw_path = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    model_path = 'deployment/model.pkl'
    feat_path = 'deployment/feature_cols.pkl'

    # Load data
    raw = load_data(raw_path)

    # Run EDA (saves plots to reports/figures)
    run_eda(raw, raw_path)

    # Preprocess
    df = preprocess_data(raw)

    # Train model and save
    X_train, X_test, y_train, y_test = train_and_save_model(df, model_path, feat_path)

    # Evaluate (saves evaluation plots)
    results = evaluate_model(model_path, feat_path, X_test, y_test)
    print('Evaluation Results:', results)

if __name__ == '__main__':
    main()
