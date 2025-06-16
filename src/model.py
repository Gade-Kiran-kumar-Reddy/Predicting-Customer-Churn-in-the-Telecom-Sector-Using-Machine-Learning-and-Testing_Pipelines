from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model(df, model_path, feat_path):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('clf',   RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, model_path)
    joblib.dump(X.columns.tolist(), feat_path)
    return X_train, X_test, y_train, y_test
