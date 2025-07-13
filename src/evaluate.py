import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

FIG_DIR = 'reports/figures'
os.makedirs(FIG_DIR, exist_ok=True)

def evaluate_model(model_path, feat_path, X_test, y_test):
    model = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, proba)
    }
    print(classification_report(y_test, preds))
    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/confusion_matrix.png', dpi=300)
    plt.close(fig)
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig2 = plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.2f}")
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    fig2.savefig(f'{FIG_DIR}/roc_curve.png', dpi=300)
    plt.close(fig2)
    return metrics



# import os
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_score, recall_score,
#     roc_auc_score, classification_report, confusion_matrix, roc_curve
# )

# FIG_DIR = 'reports/figures'
# os.makedirs(FIG_DIR, exist_ok=True)

# def evaluate_model(model_path, feat_path, X_test, y_test):
#     """
#     Load model or calibrated dict, generate predictions, compute metrics,
#     and produce confusion matrix & ROC curve figures.
#     """
#     # Load saved model or dict
#     data = joblib.load(model_path)
#     if isinstance(data, dict) and 'model' in data and 'threshold' in data:
#         clf = data['model']
#         thr = data['threshold']
#         proba = clf.predict_proba(X_test)[:, 1]
#         preds = (proba > thr).astype(int)
#     else:
#         clf = data
#         preds = clf.predict(X_test)
#         proba = clf.predict_proba(X_test)[:, 1]

#     # Compute metrics
#     metrics = {
#         'accuracy': accuracy_score(y_test, preds),
#         'f1': f1_score(y_test, preds),
#         'precision': precision_score(y_test, preds),
#         'recall': recall_score(y_test, preds),
#         'roc_auc': roc_auc_score(y_test, proba)
#     }
#     print(classification_report(y_test, preds))

#     # Confusion matrix plot
#     cm = confusion_matrix(y_test, preds)
#     fig, ax = plt.subplots(figsize=(6,5))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('Actual')
#     ax.set_title('Confusion Matrix')
#     fig.tight_layout()
#     fig.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=300)
#     plt.close(fig)

#     # ROC curve plot
#     fpr, tpr, _ = roc_curve(y_test, proba)
#     fig2, ax2 = plt.subplots(figsize=(6,5))
#     ax2.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.2f}")
#     ax2.plot([0,1], [0,1], '--', color='grey')
#     ax2.set_xlabel('False Positive Rate')
#     ax2.set_ylabel('True Positive Rate')
#     ax2.set_title('ROC Curve')
#     ax2.legend(loc='lower right')
#     fig2.tight_layout()
#     fig2.savefig(os.path.join(FIG_DIR, 'roc_curve.png'), dpi=300)
#     plt.close(fig2)

#     return metrics
