import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import preprocess_data

FIG_DIR = 'reports/figures'
os.makedirs(FIG_DIR, exist_ok=True)

def run_eda(raw_df, raw_path):
    # Raw EDA
    plt.figure(figsize=(6,4))
    sns.countplot(data=raw_df, x='Churn')
    plt.title('Churn Distribution')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/churn_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    sns.countplot(data=raw_df, x='Contract', hue='Churn', palette='pastel')
    plt.title('Contract Type vs Churn')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/contract_vs_churn.png', dpi=300)
    plt.close()

    # Processed EDA
    df = preprocess_data(raw_df.copy())
    plt.figure(figsize=(20,16))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                square=True, cbar_kws={'shrink':.5}, linewidths=.5, annot_kws={'size':8})
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/feature_correlation_heatmap.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title('Monthly Charges vs Churn')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/monthly_charges_vs_churn.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8,6))
    sns.histplot(df['tenure'], bins=30, kde=True)
    plt.title('Distribution of Tenure')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/tenure_distribution.png', dpi=300)
    plt.close()

    # Pairplot
    selected = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
    sns.pairplot(df[selected], hue='Churn', palette='Set2')
    plt.suptitle('Pairplot of Selected Features', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/pairplot.png', dpi=300)
    plt.close()
