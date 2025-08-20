import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

FIG_DIR = 'reports/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def train_and_save_best_model(df, model_path, feat_path, out_dir="deployment"):
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # split features / target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0}) if df['Churn'].dtype == object else df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # save feature columns (kept for backward-compat with your API; consider serializing the full pipeline only)
    joblib.dump(X.columns.tolist(), feat_path)

    # common preprocessing
    imputer = SimpleImputer(strategy='median')

    # define candidate models (single-threaded to avoid Windows multiprocessing crash / nested parallelism)
    candidates = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=1  # ⬅️ important on Windows
        ),
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=1  # ⬅️ important on Windows
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42
            # (LogReg doesn't use n_jobs in lbfgs)
        )
    }

    # hyperparameter spaces for optimization
    param_grids = {
        'RandomForest': {
            'clf__n_estimators': [100, 200, 500],
            'clf__max_depth': [None, 10, 20],
            'clf__min_samples_split': [2, 5]
        },
        'XGBoost': {
            'clf__n_estimators': [100, 200, 500],
            'clf__max_depth': [3, 6, 10],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        },
        'LogisticRegression': {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs', 'liblinear']
        }
    }

    results = []
    best_score = -np.inf
    best_name = None
    best_pipe = None

    # train baseline and optimized models
    for name, clf in candidates.items():
        pipe = Pipeline([('impute', imputer), ('clf', clf)])
        print(f"Training baseline {name}...")
        pipe.fit(X_train, y_train)
        base_acc = pipe.score(X_test, y_test)

        print(f"Optimizing {name}...")
        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grids[name],
            n_iter=5,
            cv=3,
            scoring='accuracy',
            n_jobs=1,        # ⬅️ Windows-safe: avoid joblib/loky multiprocessing
            random_state=42,
            verbose=1
        )
        rs.fit(X_train, y_train)
        opt_acc = rs.best_estimator_.score(X_test, y_test)

        print(f"{name}: base={base_acc:.4f}, optimized={opt_acc:.4f}")
        results.append({'model': name, 'base_accuracy': base_acc, 'opt_accuracy': opt_acc})

        # track best optimized
        if opt_acc > best_score:
            best_score = opt_acc
            best_name = name
            best_pipe = rs.best_estimator_

    print(f"Best optimized model: {best_name} with accuracy {best_score:.4f}")
    # save the best optimized pipeline
    joblib.dump(best_pipe, model_path)
    print(f"Saved optimized best pipeline to {model_path}")

    # comparison DataFrame
    df_results = pd.DataFrame(results)
    # save CSV
    csv_path = os.path.join(out_dir, 'model_comparison.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")

    # plot before vs after accuracies
    labels = df_results['model']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, df_results['base_accuracy'], width, label='Baseline')
    ax.bar(x + width / 2, df_results['opt_accuracy'], width, label='Optimized')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy: Baseline vs Optimized')
    ax.legend()
    for i, (b, o) in enumerate(zip(df_results['base_accuracy'], df_results['opt_accuracy'])):
        ax.text(i - width / 2, b + 0.01, f"{b:.2f}", ha='center')
        ax.text(i + width / 2, o + 0.01, f"{o:.2f}", ha='center')
    plt.tight_layout()
    plot_path = os.path.join(FIG_DIR, 'model_comparison.png')
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison plot to {plot_path}")

    return X_train, X_test, y_train, y_test
