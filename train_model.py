import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

def create_and_train():
    print("[*] Generating synthetic banking data (20,000 records)...")
    np.random.seed(42)
    n_samples = 20000
    
    data = {
        'Time': np.random.uniform(0, 86400, n_samples),
        'Amount': np.random.exponential(scale=100, size=n_samples),
        'V1': np.random.normal(0, 1.5, n_samples),
        'V2': np.random.normal(0, 1.5, n_samples),
        'V3': np.random.normal(0, 1.5, n_samples),
    }
    df = pd.DataFrame(data)
    
    df['Class'] = np.where((df['Amount'] > 400) & (df['V1'] < -1.0), 1, 0)
    noise = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    df['Class'] = np.maximum(df['Class'], noise)

    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = RobustScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print("[*] Training XGBoost Model...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='aucpr')
    model.fit(X_train_sm, y_train_sm)

    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, 'models/fraud_pipeline.pkl')
    print("[+] Model saved successfully to models/fraud_pipeline.pkl")

if __name__ == "__main__":
    create_and_train()
