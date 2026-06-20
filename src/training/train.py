import sys
sys.path.append(".")

import os
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from src.processing.feature_engineer import load_raw_csv, engineer_features

# Load and combine all classes
dfs = []
for label in ["safe", "medium_risk", "high_risk"]:
    df = load_raw_csv(f"data/raw/{label}.csv")
    dfs.append(engineer_features(df))

data = pd.concat(dfs, ignore_index=True)

# Encode labels
le = LabelEncoder()
data["label_enc"] = le.fit_transform(data["label"])

X = data.drop(columns=["label", "label_enc"])
y = data["label_enc"]

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# MLflow tracking
mlflow.set_experiment("fall-risk-detection")

with mlflow.start_run():
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("accuracy", acc)
    mlflow.xgboost.log_model(model, "model")

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model locally
    os.makedirs("models", exist_ok=True)
    model.save_model("models/fall_risk_model.json")
    print("\nModel saved to models/fall_risk_model.json")

