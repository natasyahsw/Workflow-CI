"""
Model Training Script untuk CI/CD Pipeline
===========================================
Script ini dijalankan oleh GitHub Actions workflow di dalam folder MLProject.
Melakukan training model dan logging ke MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings

warnings.filterwarnings("ignore")

# ===================================================================
# KONFIGURASI
# ===================================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "Credit Scoring"
RANDOM_STATE = 42

# ===================================================================
# SETUP
# ===================================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"[CI] MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"[CI] Experiment: {EXPERIMENT_NAME}")


def load_and_preprocess():
    """Load dan preprocess dataset."""
    print("[CI] Loading dataset...")

    # Coba load dari preprocessed data terlebih dahulu
    prep_dir = "credit_risk_preprocessing"
    if os.path.exists(os.path.join(prep_dir, "X_train.csv")):
        print(f"[CI] Loading preprocessed data dari {prep_dir}/")
        X_train = pd.read_csv(os.path.join(prep_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(prep_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(prep_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(prep_dir, "y_test.csv")).values.ravel()
        return X_train, X_test, y_train, y_test

    # Fallback: load raw dan preprocess
    try:
        if os.path.exists("credit_risk_dataset.csv"):
            df = pd.read_csv("credit_risk_dataset.csv")
        else:
            df = pd.read_csv("../credit_risk_dataset.csv")
        print(f"[CI] Raw dataset loaded: {df.shape}")
    except Exception as e:
        print(f"[CI] Error loading dataset: {e}")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=11, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(11)])
        df["loan_status"] = y
        print(f"[CI] Using synthetic data: {df.shape}")

    target_col = "loan_status"

    # Preprocessing
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Save preprocessed
    os.makedirs(prep_dir, exist_ok=True)
    X_train.to_csv(os.path.join(prep_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(prep_dir, "X_test.csv"), index=False)
    y_train_s = pd.Series(y_train)
    y_test_s = pd.Series(y_test)
    y_train_s.to_csv(os.path.join(prep_dir, "y_train.csv"), index=False)
    y_test_s.to_csv(os.path.join(prep_dir, "y_test.csv"), index=False)

    print(f"[CI] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train():
    """Train model dan log ke MLflow."""
    X_train, X_test, y_train, y_test = load_and_preprocess()

    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE,
    }

    with mlflow.start_run(run_name="CI-Pipeline-Run") as run:
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, "model", registered_model_name="credit-scoring")

        # Log confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig("training_confusion_matrix.png", dpi=150)
        mlflow.log_artifact("training_confusion_matrix.png")
        plt.close()

        with open("metric_info.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        # Save model.pkl for Docker
        import joblib
        joblib.dump(model, "model.pkl")

        mlflow.set_tag("pipeline", "CI")
        mlflow.set_tag("trigger", os.getenv("GITHUB_EVENT_NAME", "manual"))

        print(f"\n[CI] ===== TRAINING COMPLETE =====")
        print(f"[CI] Run ID: {run.info.run_id}")
        for k, v in metrics.items():
            print(f"[CI] {k}: {v:.4f}")

    return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    print(f"\n[CI] Latest Run ID: {run_id}")
