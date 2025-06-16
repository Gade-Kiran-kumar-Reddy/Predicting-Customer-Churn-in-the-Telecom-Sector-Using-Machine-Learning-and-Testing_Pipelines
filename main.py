# main.py
import os
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.eda import run_eda
from src.model import train_and_save_best_model
from src.evaluate import evaluate_model

def main():
    os.makedirs('deployment', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    raw_path   = os.path.join('data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    model_path = os.path.join('deployment', 'model.pkl')
    feat_path  = os.path.join('deployment', 'feature_cols.pkl')

    raw = load_data(raw_path)
    run_eda(raw, raw_path)
    df = preprocess_data(raw)

    X_train, X_test, y_train, y_test = train_and_save_best_model(
        df,
        model_path=model_path,
        feat_path=feat_path,
        out_dir='deployment'
    )
    results = evaluate_model(model_path, feat_path, X_test, y_test)
    print('Final evaluation on best model:', results)

if __name__ == '__main__':
    main()
