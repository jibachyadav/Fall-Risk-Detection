import numpy as np
import pandas as pd

LANDMARK_COLUMNS = [f"lm_{i}_{axis}" for i in range(33) for axis in ["x", "y", "z", "vis"]]

def load_raw_csv(filepath):
    cols = LANDMARK_COLUMNS + ["label"]
    df = pd.read_csv(filepath, header=None, names=cols)
    return df

def get_landmark(df, index):
    x = df[f"lm_{index}_x"]
    y = df[f"lm_{index}_y"]
    return np.array([x, y]).T

def calculate_angle(a, b, c):
    """Angle at point b, formed by a-b-c."""
    ba = a - b
    bc = c - b
    cosine = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def engineer_features(df):
    features = pd.DataFrame()

    # Spine angle: shoulder midpoint - hip midpoint - knee midpoint
    left_shoulder = get_landmark(df, 11)
    right_shoulder = get_landmark(df, 12)
    left_hip = get_landmark(df, 23)
    right_hip = get_landmark(df, 24)
    left_knee = get_landmark(df, 25)
    right_knee = get_landmark(df, 26)

    shoulder_mid = (left_shoulder + right_shoulder) / 2
    hip_mid = (left_hip + right_hip) / 2
    knee_mid = (left_knee + right_knee) / 2

    features["spine_angle"] = calculate_angle(shoulder_mid, hip_mid, knee_mid)

    # Lateral lean: horizontal distance between shoulder mid and hip mid
    features["lateral_lean"] = np.abs(shoulder_mid[:, 0] - hip_mid[:, 0])

    # Left and right knee angles
    left_ankle = get_landmark(df, 27)
    right_ankle = get_landmark(df, 28)
    features["left_knee_angle"] = calculate_angle(left_hip, left_knee, left_ankle)
    features["right_knee_angle"] = calculate_angle(right_hip, right_knee, right_ankle)

    # Visibility scores of key joints (confidence)
    features["hip_visibility"] = (df["lm_23_vis"] + df["lm_24_vis"]) / 2
    features["knee_visibility"] = (df["lm_25_vis"] + df["lm_26_vis"]) / 2

    features["label"] = df["label"].values
    return features
