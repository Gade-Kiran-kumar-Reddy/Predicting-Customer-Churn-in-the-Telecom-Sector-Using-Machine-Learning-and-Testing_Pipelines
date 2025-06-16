from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os

def train_and_save_best_model(df, model_path, feat_path, out_dir="deployment"):
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # split features / target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # save feature columns once
    joblib.dump(X.columns.tolist(), feat_path)

    # common preprocessing
    imputer = SimpleImputer(strategy='median')

    # define candidate models
    candidates = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgb': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ),
        'lr': LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ),
    }

    best_score = -1
    best_name = None
    best_pipe = None

    # train, evaluate, track best
    for name, clf in candidates.items():
        pipe = Pipeline([
            ('impute', imputer),
            ('clf', clf)
        ])
        print(f"Training {name} pipeline...")
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)  # accuracy on test set
        print(f" → {name} accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe

    # save the best
    print(f"Best model: {best_name} with accuracy {best_score:.4f}")
    joblib.dump(best_pipe, model_path)
    print(f"Saved best pipeline to {model_path}")

    return X_train, X_test, y_train, y_test
