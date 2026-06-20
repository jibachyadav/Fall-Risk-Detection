import sys
sys.path.append(".")

import cv2
import time
import requests
import streamlit as st
import mediapipe as mp
import numpy as np
from src.ingestion.pose_extractor import extract_pose_landmarks

API_URL = "http://127.0.0.1:8001/predict"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Fall Risk Detection", layout="wide")
st.title("Hospital Patient Fall Risk Detection")

RISK_COLORS = {
    "safe": "green",
    "medium_risk": "orange",
    "high_risk": "red"
}

run = st.toggle("Start Live Detection")
frame_placeholder = st.empty()
result_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose() as pose_model:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read from webcam")
                break

            landmarks = extract_pose_landmarks(frame, pose_model)

            if landmarks is not None:
                # Draw skeleton on frame
                results = pose_model.process(frame[:, :, ::-1])
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Build features and call API
                spine_angle = float(np.degrees(np.arccos(np.clip(
                    np.dot(landmarks[11, :2] - landmarks[23, :2],
                           landmarks[25, :2] - landmarks[23, :2]) /
                    (np.linalg.norm(landmarks[11, :2] - landmarks[23, :2]) *
                     np.linalg.norm(landmarks[25, :2] - landmarks[23, :2]) + 1e-6),
                    -1, 1))))

                payload = {
                    "spine_angle": spine_angle,
                    "lateral_lean": float(abs(landmarks[11, 0] - landmarks[23, 0])),
                    "left_knee_angle": 170.0,
                    "right_knee_angle": 170.0,
                    "hip_visibility": float((landmarks[23, 3] + landmarks[24, 3]) / 2),
                    "knee_visibility": float((landmarks[25, 3] + landmarks[26, 3]) / 2)
                }

                try:
                    response = requests.post(API_URL, json=payload, timeout=1)
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    color = RISK_COLORS[prediction]
                    result_placeholder.markdown(
                        f"### Risk Level: :{color}[{prediction.upper()}] "
                        f"(confidence: {confidence:.2%})"
                    )
                except Exception:
                    result_placeholder.warning("API not reachable")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

