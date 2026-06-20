import sys
sys.path.append(".")

from src.processing.feature_engineer import load_raw_csv, engineer_features

for label in ["safe", "medium_risk", "high_risk"]:
    df = load_raw_csv(f"data/raw/{label}.csv")
    features = engineer_features(df)
    print(f"\n--- {label} ---")
    print(features.drop(columns=["label"]).describe().round(2))

